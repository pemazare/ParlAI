#!/usr/bin/env python3
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn._functions.rnn import LSTMCell
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

# from fairseq.modules import ConvTBC

# from line_profiler import LineProfiler
# profile = LineProfiler()

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

def initial_shift_offset(shift, di_size):
    return 0 if shift >= 0 else (di_size - 1) * abs(shift)

def shift(x, shift, di, dj, padding):
    di_size, dj_size = x.size(di), x.size(dj)
    if di_size == 0:
        return x
    output_size = list(x.size())
    output_size[dj] = dj_size + abs(shift) * (di_size - 1)
    output = x.new(*output_size).fill_(padding)
    offset = initial_shift_offset(shift, di_size)
    for i in range(di_size):
        output.narrow(dj, offset, dj_size).select(di, i).copy_(x.select(di, i))
        offset += shift
    return output

def unshift(x, shift, di, dj):
    di_size, output_dj_size = x.size(di), x.size(dj)
    if di_size == 0:
        return x
    dj_size = output_dj_size - abs(shift) * (di_size - 1)
    output_size = list(x.size())
    output_size[dj] = dj_size
    output = x.new(*output_size)
    offset = initial_shift_offset(shift, di_size)
    for i in range(di_size):
        output.select(di, i).copy_(x.narrow(dj, offset, dj_size).select(di, i))
        offset += shift
    return output

class DiagonalUnshift(Function):
    """Reverses the DiagonalShift operation"""
    def __init__(self, shift, di, dj):
        self.shift = shift
        self.di = di
        self.dj = dj

    def forward(self, x):
        return unshift(x, self.shift, self.di, self.dj)

    def backward(self, grad_output):
        return shift(grad_output, self.shift, self.di, self.dj, 0)


class DiagonalShift(Function):
    """Transforms an input by shifting each block relative to the previous block
    The shifting plane is defined by two dimensions: di (dimension defining the
    blocks) and dj (shifting direction)
    The width of the shift can be controlled via the "shift" parameter.
    The output tensor is padded with the given value

    Input: tensor of size d1 x ... x di x ... x dj x ... x dn
    Output: tensor of size d1 x ... x di x ... x (dj + abs(shift) * (di - 1)) x ... x dn

    Examples with 2d tensor, di=0 and dj=1:

    * shift=1:

       + + + + + + +        + + + + + + + 0 0
       + + + + + + +   =>   0 + + + + + + + 0
       + + + + + + +        0 0 + + + + + + +

    * shift=2:

       + + + + + + +        + + + + + + + 0 0 0 0
       + + + + + + +   =>   0 0 + + + + + + + 0 0
       + + + + + + +        0 0 0 0 + + + + + + +

    * shift=-1:

       + + + + + + +        0 0 + + + + + + +
       + + + + + + +   =>   0 + + + + + + + 0
       + + + + + + +        + + + + + + + 0 0
    """
    def __init__(self, shift=1, di=0, dj=1, padding=0):
        super()
        self.shift = shift
        self.di = di
        self.dj = dj
        self.padding = padding

    def forward(self, x):
        return shift(x, self.shift, self.di, self.dj, self.padding)

    def backward(self, grad_output):
        return unshift(grad_output, self.shift, self.di, self.dj)

    def unshift(self):
        return DiagonalUnshift(self.shift, self.di, self.dj)


class GridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, single_direction=False, horizontal_only=False):
        super(GridLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.single_direction = single_direction
        self.horizontal_only = horizontal_only
        self.num_directions = 1 if self.single_direction else 4

        self.x_conv = nn.ModuleList()
        self.h_conv = nn.ModuleList()
        # for _ in range(self.num_directions):
        #     self.x_conv.append(ConvTBC(input_size, 4 * hidden_size, 1))
        #     self.h_conv.append(ConvTBC(hidden_size, 4 * hidden_size, 2, padding=1))

        self.x_bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.h_bias = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in (self.h_bias, self.x_bias):
            weight.data.uniform_(-stdv, stdv)
            # increase forget gate bias
            weight.data[self.hidden_size:2 * self.hidden_size] += 1
        for convs in (self.x_conv, self.h_conv):
            for conv in convs:
                n = conv.in_channels
                for k in conv.kernel_size:
                    n *= k
                stdv = 1.0 / math.sqrt(n)
                conv.weight.data.uniform_(-stdv, stdv)
                conv.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: [c_size, q_size, batch_size, input_size]
        h_all_outputs = []
        c_all_outputs = []
        c_size, q_size, batch_size, input_size = x.size()
        num_parallel_items = q_size * batch_size
        c_padding = Variable(x.data.new(1, batch_size, self.hidden_size).zero_())
        initial_c = Variable(x.data.new(q_size, batch_size, self.hidden_size).zero_())
        initial_h = Variable(x.data.new(q_size, batch_size, self.hidden_size).zero_())
        pos_shift = DiagonalShift(1, 1, 0)
        pos_unshift = pos_shift.unshift()
        pos_shifted_x = pos_shift(x)
        if self.num_directions > 0:
            neg_shift = DiagonalShift(-1, 1, 0)
            neg_unshift = neg_shift.unshift()
            neg_shifted_x = neg_shift(x)
        for d in range(self.num_directions):
            q_reverse = d == 1 or d == 3
            c_reverse = d == 2 or d == 3
            shifted_x = pos_shifted_x if (q_reverse == c_reverse) else neg_shifted_x
            shifted_c_size = shifted_x.size(0)
            for is_vertical in (False,) if self.horizontal_only else (False, True):
                h_outputs = []
                current_c = initial_c
                current_h = initial_h
                for t in range(shifted_c_size):
                    wx = self.x_conv[d](shifted_x[-(t + 1) if c_reverse else t])
                    wh = self.h_conv[d](current_h).narrow(0, 0 if (q_reverse == c_reverse) else 1, q_size)
                    if is_vertical:
                        # shift current_c to propagate vertically instead of horizontally
                        if q_reverse == c_reverse:
                            current_c = torch.cat([c_padding, current_c[:-1]])
                        else:
                            current_c = torch.cat([current_c[1:], c_padding])
                    # wh, wx, current_c = map(lambda x: x.transpose(1, 2).contiguous().view(batch_size * q_size, -1), (wh, wx, current_c))
                    wx = wx.view(num_parallel_items, -1)
                    wh = wh.view(num_parallel_items, -1)
                    state = fusedBackend.LSTMFused.apply
                    current_h, current_c = state(wx, wh, current_c, self.x_bias, self.h_bias)
                    current_h = current_h.view(q_size, batch_size, -1)
                    current_c = current_c.view(q_size, batch_size, -1)
                    # current_c, current_h = map(lambda x: x.view(batch_size, q_size, -1).contiguous().transpose(1, 2), (current_c, current_h))
                    h_outputs.append(current_h)
                if c_reverse:
                    h_outputs.reverse()
                unshift = pos_unshift if (q_reverse == c_reverse) else neg_unshift
                h_all_outputs.append(unshift(torch.stack(h_outputs)))
        return torch.cat(h_all_outputs, dim=3)


class StackedCNN(nn.Module):
    def __init__(self, input_size, num_filters, num_layers, **kwargs):

        super(StackedCNN, self).__init__()
        self.kwargs = kwargs
        self.input_size = input_size
        self.num_filters = num_filters  # list of num_filters of each kernel width
        self.num_layers = num_layers
        self.cnns = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.word_dropout = torch.nn.Dropout2d(p=kwargs['dropout_rate'])
        self.dropout_2d_filter = torch.nn.Dropout2d(p=kwargs['dropout_rate'])
        assert not (
            kwargs['qemb_match_all_layers'] and kwargs['qemb_match_last_layer']
        )
        self.output_size_each_layer = sum(num_filters)
        self.output_size = num_layers * self.output_size_each_layer if kwargs[
            'concat_layers'
        ] else self.output_size_each_layer
        if kwargs['qemb_match_first_layer'] or kwargs['qemb_match_all_layers'] or kwargs['qemb_match_last_layer']:
            if kwargs['attn_type'] == 'mlp':
                self.qemb_match = SeqMLPAttnMatch(
                    q_hidden_size=kwargs['question_hidden_size'],
                    p_hidden_size=kwargs['output_size_each_layer'],
                    output_size=kwargs['output_size_each_layer']
                )
            else:
                assert kwargs['question_hidden_size'
                             ] == self.output_size_each_layer
                self.qemb_match = SeqAttnMatchCNN()
        if kwargs['qemb_match_last_layer']:
            kwargs['share_gates_param'] = True
        if kwargs['use_gated_question_attention']:
            self.gate_params = Projection(
                kwargs['question_hidden_size'], kwargs['question_hidden_size'],
                kwargs['dropout_rate']
            ) if kwargs['share_gates_param'] else nn.ModuleList()

        for l in range(num_layers):
            dilation = 2 if kwargs['use_dilation'] else 1
            layer_cnns = nn.ModuleList()  # cnns for a layer
            input_size = input_size if l == 0 else self.output_size_each_layer
            self.projections.append(
                Projection(
                    input_size, self.output_size_each_layer,
                    kwargs['dropout_rate']
                ) if input_size != self.output_size_each_layer else None
            )
            if l > 0 and (
                kwargs['qemb_match_all_layers'] or
                kwargs['qemb_match_last_layer']
            ) and not kwargs['use_gated_question_attention']:
                input_size *= 2  # if doing question attention, but not doing
                # gates, then we just concat and hence the size doubles
            for kernel_counter, k in enumerate(kwargs['kernel_widths']):
                # if the kernal width isnt odd, I cant pad to get the same number
                # of outputs as inputs
                assert k % 2 == 1
                padding = (dilation * (k - 1)) // 2  # assuming stride = 1
                num_filters = self.num_filters[kernel_counter]
                if kwargs['use_gated_cnns']:
                    num_filters *= 2
                layer_cnns.append(
                    Conv1d(
                        in_channels=input_size,
                        out_channels=num_filters,
                        kernel_size=k,
                        dropout=kwargs['dropout_rate'],
                        padding=padding,
                        dilation=dilation,
                    )
                )
            self.cnns.append(layer_cnns)
            if l > 0 and kwargs['use_gated_question_attention'] \
            and not kwargs['share_gates_param']:
                self.gate_params.append(
                    Projection(
                        kwargs['question_hidden_size'],
                        kwargs['question_hidden_size'], kwargs['dropout_rate']
                    )
                )

    def forward(self, x, x_mask, question_hiddens=None, question_mask=None, question_emb=None):
        """
        @param x: sequence of word embeddings. [B, seq_len, embedding_dim]
        @param x_mask: word masks [B, seq_len]
        @param question_hiddens: question hidden vectors [B, q_seq_len, embedding_dim]
        @param question_mask: [B, q_seq_len]
        """
        # Transpose batch and sequence dims
        if self.kwargs['use_word_dropout']:
            x = x.unsqueeze(3)  # [B, seq_len, embedding_dim, 1]
            x = self.word_dropout(x)
            x = x.squeeze(3)  # [B, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # (B, embdding_dim, seq_len)
        # [B, embdding_dim, q_seq_len]
        question_hiddens = question_hiddens.transpose(1, 2)
        if self.kwargs['qemb_match_first_layer']:
            question_emb = question_emb.transpose(1, 2)
            input_no_features = x.narrow(1, 0, question_emb.size()[1])
            # first calculate the attention
            q_weighted_emb = self.qemb_match(
                input_no_features, question_emb, question_mask
            )  # [B, seq_len, embedding_dim]
            x = torch.cat([x, q_weighted_emb], 1)

        inp = x
        outputs = [x]
        for i in range(self.num_layers):
            inp = outputs[-1]
            if i > 0 and self.kwargs['qemb_match_all_layers']:
                if self.kwargs['use_gated_question_attention'] and self.kwargs['qemb_project_before_softmax']:
                    gate_params = self.gate_params if self.kwargs[
                        'share_gates_param'
                    ] else self.gate_params[i - 1]
                    inp = gate_params(inp)

                # first calculate the attention
                q_weighted_emb = self.qemb_match(
                    inp, question_hiddens, question_mask
                )  # [B, seq_len, output_size_each_layer]
                if self.kwargs['use_gated_question_attention'] and not self.kwargs['qemb_project_before_softmax']:
                    # transform the q_weighted_emb
                    gate_params = self.gate_params if self.kwargs[
                        'share_gates_param'
                    ] else self.gate_params[i - 1]
                    q_weighted_emb = gate_params(q_weighted_emb)
                # concatentate q_weighted_emb with inp
                inp = torch.cat([inp, q_weighted_emb],
                                1)  # [B, 2*output_size_each_layer, seq_len]
                # linear mapping with the gate params and then sigmoid
                if self.kwargs['use_gated_question_attention']:
                    inp = F.glu(inp, dim=1)
            # Apply dropout to hidden input
            if self.kwargs['dropout_rate'] > 0:
                if self.kwargs['use_dropout_2d_filter']:
                    inp = (self.dropout_2d_filter(inp.unsqueeze(3))).squeeze(3)
                elif self.kwargs['use_word_dropout_all_layers']:
                    inp = inp.unsqueeze(3)  # [B, seq_len, embedding_dim, 1]
                    inp = self.word_dropout(inp)
                    inp = inp.squeeze(3)  # [B, seq_len, embedding_dim]
                else:
                    inp = F.dropout(
                        inp, p=self.dropout_rate, training=self.training
                    )
            all_out_k = []
            for k in range(len(self.kwargs['kernel_widths'])):
                # num_filters_k is number of filters with kernel_width kernel_width[k]
                out_k = self.cnns[i][k](inp)
                if self.kwargs['use_gated_cnns']:
                    out_k = F.glu(out_k, dim=1)
                else:
                    out_k = F.relu(out_k)
                all_out_k.append(out_k)
            # concatenate the output of all filters
            out = torch.cat(
                all_out_k, 1
            )  # (B, output_size_each_layer, seq_len)
            if self.kwargs['use_residual_layer']:
                residual = outputs[-1] if self.projections[i] is None else self.projections[i](outputs[-1])
                out = (out + residual) * math.sqrt(0.5)
            outputs.append(out)
        if self.kwargs['qemb_match_last_layer']:
            last_layer_output = outputs[-1]
            if self.kwargs['use_gated_question_attention'] and self.kwargs['qemb_project_before_softmax']:
                # transform the q_weighted_emb
                gate_params = self.gate_params
                last_layer_output = gate_params(last_layer_output)

            q_weighted_emb = self.qemb_match(
                last_layer_output, question_hiddens, question_mask
            )  # [B, seq_len, output_size_each_layer]
            if self.kwargs['use_gated_question_attention'] and not self.kwargs['qemb_project_before_softmax']:
                # transform the q_weighted_emb
                gate_params = self.gate_params
                q_weighted_emb = gate_params(q_weighted_emb)
            last_layer_output = torch.cat(
                [last_layer_output, q_weighted_emb], 1
            )
            if self.kwargs['use_gated_question_attention']:
                last_layer_output = F.glu(last_layer_output, dim=1)
            outputs[-1] = last_layer_output  # replace last layer
        if self.kwargs['concat_layers']:
            output = torch.cat(
                outputs[1:], 1
            )  # [B, num_layes*output_size_each_layer, seq_len]
        else:
            output = outputs[-1]  # [B, seq_len, num_filters]
        # Dropout on output layer
        if self.kwargs['dropout_output'] and self.kwargs['dropout_rate'] > 0:
            output = F.dropout(
                output, p=self.kwargs['dropout_rate'], training=self.training
            )
        # profile.print_stats()
        return output.transpose(1, 2)


class StackedBRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout_rate=0,
        dropout_output=False,
        rnn_type=nn.LSTM,
        concat_layers=False,
        padding=False,
        use_word_dropout=False
    ):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(
                rnn_type(
                    input_size, hidden_size, num_layers=1, bidirectional=True
                )
            )

    def forward(self, x, x_mask, question_hiddens=None, question_mask=None, question_emb=None):
        """
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """
        Faster encoding that ignores any padding.
        """
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(
                    rnn_input, p=self.dropout_rate, training=self.training
                )
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(
                output, p=self.dropout_rate, training=self.training
            )
        return output

    def _forward_padded(self, x, x_mask):
        """
        Slower (significantly), but more precise, encoding that handles padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(
                    rnn_input.data, p=self.dropout_rate, training=self.training
                )
                rnn_input = nn.utils.rnn.PackedSequence(
                    dropout_input, rnn_input.batch_sizes
                )
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        if output.size(1) != x.size(0):
            # Add padding to recover initial dimensions
            output_pad_size = [x for x in output.size()]
            output_pad_size[1] = x.size(0) - output.size(1)
            output = torch.cat(
                [output, output.data.new(*output_pad_size).zero_()], dim=1
            )

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(
                output, p=self.dropout_rate, training=self.training
            )
        return output


class SeqMLPAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(MLP(y_j * x_i))
    """

    def __init__(self, q_hidden_size, p_hidden_size, output_size):
        super(SeqMLPAttnMatch, self).__init__()
        self.q_linear = nn.Linear(q_hidden_size, output_size)
        self.p_linear = nn.Linear(p_hidden_size, output_size)
        self.last_linear = nn.Linear(output_size, 1)
        self.q_hidden_size = q_hidden_size
        self.p_hidden_size = p_hidden_size
        self.output_size = output_size

    def forward(self, x, y, y_mask):
        """
        Input shapes:
            x = batch * len1 * q_hidden_size
            y = batch * len2 * p_hidden_size
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * output_size
        """
        # take linear transformations
        x_proj = self.p_linear(x.view(-1, x.size(2))).view(
            x.size(0), x.size(1), self.output_size
        )  # [B, len1, output_size]
        y_proj = self.q_linear(y.view(-1, y.size(2))).view(
            y.size(0), y.size(1), self.output_size
        )  # [B, len2, output_size]

        x_proj = torch.unsqueeze(x_proj, 2).expand(
            x.size(0), x.size(1), y.size(1), self.output_size
        )  # [B, len1, len2, output_size]
        y_proj = torch.unsqueeze(y_proj, 1).expand(
            y.size(0), x.size(1), y.size(1), self.output_size
        )  # [B, len1, len2, output_size]
        mlp_out = F.tanh(x_proj + y_proj).view(
            -1, self.output_size
        )  # [(B*len1*len2), output_size]
        scores = torch.squeeze(
            self.last_linear(mlp_out).view(-1, x.size(1), y.size(1)), 2
        )  # [B, len1, len2)]
        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))  # [B*len1, len2]
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))  # [B, len1, len2]

        # Take weighted average
        matched_seq = alpha.bmm(y)

        return matched_seq


class SeqSelfAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False, proj_size=None, extra_scaling=None, remove_id=False):
        super(SeqSelfAttnMatch, self).__init__()
        self.extra_scaling = extra_scaling or 1
        self.remove_id = remove_id
        if not identity:
            self.proj_size = proj_size or input_size
            self.linear = nn.Linear(input_size, self.proj_size)
        else:
            self.linear = None

    def forward(self, x, x_mask, scale_dot_product=False, return_projected=False, residual_attention=False):
        """
        Input shapes:
            x = batch * len * h
            x_mask = batch * len
        Output shapes:
            matched_seq = batch * len * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1), self.proj_size)
            x_proj = F.relu(x_proj)
        else:
            x_proj = x
        # Compute scores
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        # Mask padding
        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        if self.remove_id:
            mask = torch.eye(x_proj.size(1)).byte().unsqueeze(0).expand_as(scores).contiguous()
            if scores.is_cuda:
                mask = mask.cuda(async=True)
            scores.data.masked_fill_(mask, -float('inf'))

        if scale_dot_product:
            scaling = math.sqrt(x.size(2)) * self.extra_scaling
            scores.div_(scaling)


        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, x.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), x.size(1))
        # print("Alpha: min: ", alpha.min(), ", max: ", alpha.max(), ", avg abs: ", alpha.norm(1) / nel, ", size: ", alpha.size())

        # in some cases entire rows may have been wiped out by the mask above
        # in that case, the softmax will output NaN, but we'd rather have 0 in
        # order to keep the rest of the model happy.
        alpha.data.masked_fill_(x_mask.data, 0)
        if self.remove_id:
            alpha.data.masked_fill_(mask, 0)
        #
        # Take weighted average
        if return_projected:
            matched_seq = alpha.bmm(x_proj)
        else:
            matched_seq = alpha.bmm(x)
            if (matched_seq != matched_seq).data.max():
                import pdb
                pdb.set_trace()
                print(matched_seq)
            if self.remove_id:
                matched_seq = torch.cat([matched_seq, x], dim=2)
            if residual_attention:
                matched_seq += x

        return matched_seq


class SeqAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask, scale_dot_product=False, return_projected=False):
        """
        Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        if scale_dot_product:
            scores.div_(math.sqrt(x.size(2)))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # in some cases entire rows may have been wiped out by the mask above
        # in that case, the softmax will output NaN, but we'd rather have 0 in
        # order to keep the rest of the model happy.
        alpha.data.masked_fill_(y_mask.data, 0)

        # Take weighted average
        matched_seq = alpha.bmm(y)

        return matched_seq


class SeqAttnMatchCNN(nn.Module):
    """
    The difference between this class and the above one is that the shape
    of the input is [batch_size, dimension, seq_length]. Since contiguous()
    is making things slow, we can only implement dot product attention and not
    the linear proj. This is because to take the linear projs, we will have
    to do view() after taking transpose, but that will break since transpose
    makes things non-contiguous.
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self):
        super(SeqAttnMatchCNN, self).__init__()

    # @profile
    def forward(self, x, y, y_mask):
        """
        Input shapes:
            x = batch * h * len1
            y = batch * h * len2
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * h * len1
        """
        # Compute scores
        scores = x.transpose(2, 1).bmm(y)  # [B, len1, len2]
        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(2)))  # [B * len1, len2]
        alpha = alpha_flat.view(-1, x.size(2), y.size(2))  # [B, len1, len2]

        # Take weighted average
        matched_seq = alpha.bmm(y.transpose(2, 1)).transpose(
            2, 1
        )  # [B, h, len1]
        # profile.print_stats()
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """
    Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class MaxPoolOverTime(nn.Module):
    """
    Kim 2014 style max pooling over time for individual token embeddings
    """

    def __init__(self):
        super(MaxPoolOverTime, self).__init__()

    def forward(self, x, x_mask):
        # Transpose batch and sequence dims
        x = x.transpose(1, 2)  # (B, embdding_dim, seq_len)
        seq_len = x.size(2)
        m = torch.nn.MaxPool1d(kernel_size=seq_len)
        out = m(x)
        out = torch.squeeze(out, 2)
        return out


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """
    Return uniform weights over non-masked input.
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """
    x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Projection(in_features, out_features, dropout=0):
    """Weight-normalized Linear via 1x1 convolution (input: N x C x T)"""
    m = nn.Conv1d(in_features, out_features, kernel_size=1)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

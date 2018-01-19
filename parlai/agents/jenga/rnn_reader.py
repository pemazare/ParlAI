#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
from . import layers
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.parallel

from torch.autograd import Variable
import pdb

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

# ------------------------------------------------------------------------------
# Net
# ------------------------------------------------------------------------------

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

def grad_multiply(x, scale):
    return GradMultiply.apply(x, scale)


class EnsembleDocReader(nn.Module):
    def __init__(self, models, reference):
        super(EnsembleDocReader, self).__init__()
        self.networks = nn.ModuleList([model.network for model in models])
        for model, network in zip(models, self.networks):
            network.permutation = Variable(torch.cuda.LongTensor([model.word_dict[reference.word_dict[i]] for i in range(len(model.word_dict))]))

    def check_permutations_required(self):
        result = []
        for net in self.networks:
            if not hasattr(net, 'permutation_required'):
                net.permutation_required = not net.permutation.data.equal(torch.arange(net.permutation.data.size(0)).long().cuda())
            result.append(net.permutation_required)
        return result

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        raise NotImplementedError


class ResnetDocReader(nn.Module):
    def __init__(self, args, normalize=True):

        super(ResnetDocReader, self).__init__()
        # Store config
        self.args = args
        self.hidden_size = args.hidden_size
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0
        )

        # ...maybe keep them fixed
        if args.fix_embeddings:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # ...or maybe register a buffer to fill later for keeping *some* fixed
        if args.tune_partial > 0:
            buffer_size = torch.Size(
                (args.vocab_size - args.tune_partial - 2, args.embedding_dim)
            )
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        self.resnet = resnet.resnet18(
            input_size=2 * args.embedding_dim,
            blocks=args.blocks,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1) # [batch * len_d * emb_size]
        x2_emb = self.embedding(x2) # [batch * len_q * emb_size]

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(
                x1_emb, p=self.args.dropout_emb, training=self.training
            )
            x2_emb = nn.functional.dropout(
                x2_emb, p=self.args.dropout_emb, training=self.training
            )

        # Build "image"
        x_size = [x1.size(0), x1_emb.size(2), x1.size(1), x2.size(1)]
        x_img = torch.cat(
            [
                x1_emb.transpose(1, 2).unsqueeze(3).expand(*x_size),
                x2_emb.transpose(1, 2).unsqueeze(2).expand(*x_size),
            ],
            dim=1,
        )  # [batch * (2 * emb_size) * len_d * len_q]

        out = self.resnet(x_img)

        start_scores = out[:, :, 0]
        end_scores = out[:, :, 1]

        return start_scores, end_scores


class AttentionalDocReader(nn.Module):
    def __init__(self, args):
        super(AttentionalDocReader, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)

        self.question_rnns = nn.ModuleList()
        self.paragraph_rnns = nn.ModuleList()
        self.q_mlps = nn.ModuleList()
        self.p_mlps = nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_emb)
        def make_rnn(input_size):
            return nn.LSTM(input_size, args.hidden_size, num_layers=1, bidirectional=True)
        def make_mlp(input_size):
            return nn.Sequential(
                  nn.Linear(2 * input_size, input_size),
                  nn.ReLU(),
                  nn.Linear(input_size, input_size),
                  nn.ReLU()
                )

        for i in range(args.blocks):
            input_size = args.embedding_dim if i == 0 else 2 * args.hidden_size
            self.q_mlps.append(make_mlp(input_size))
            self.p_mlps.append(make_mlp(input_size))
            self.question_rnns.append(make_rnn(input_size))
            self.paragraph_rnns.append(make_rnn(input_size))
        self.fc1 = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, 2)

    def forward(self, ex):
        x1 = ex['x1']
        x1_f = ex['x1_f']
        x1_mask = ex['x1_mask']
        x2 = ex['x2']
        x2_mask = ex['x2_mask']
        # Embed both document and question

        batch_size = x1.size(0)
        paragraph_repr = self.embedding(x1) # [batch_size, p_len, emb_size]
        question_repr = self.embedding(x2) # [batch_size, q_len, emb_size]

        paragraph_repr = self.emb_dropout(paragraph_repr.unsqueeze(-1)).squeeze(-1)
        question_repr = self.emb_dropout(question_repr.unsqueeze(-1)).squeeze(-1)

        def apply_rnn(rnn, repr):
            out = rnn(repr)[0]
            out = F.dropout(out, p=self.args.dropout_rnn, training=self.training)
            return out

        def cross_repr(paragraph_repr, question_repr, par_keys, q_keys):
            alpha = par_keys.bmm(q_keys.transpose(1, 2)) # [batch_size, p_len, q_len]

            # paragraph_mask = x1_mask.unsqueeze(2).expand_as(alpha)
            # paragraph_alpha = alpha.clone()
            # paragraph_alpha.data[paragraph_mask.data] = -float('inf')
            out_paragraph = F.softmax(alpha, dim=2).bmm(question_repr)

            # question_mask = x2_mask.unsqueeze(1).expand_as(alpha)
            # question_alpha = alpha.transpose(1, 2).clone()
            # question_alpha.data[question_mask.data] = -float('inf')
            out_question = F.softmax(alpha.transpose(1, 2), dim=2).bmm(paragraph_repr)
            return out_paragraph, out_question

        for i in range(self.args.blocks):
            out_paragraph, out_question = cross_repr(paragraph_repr, question_repr,
                                                    paragraph_repr,
                                                    question_repr)
            paragraph_repr = torch.cat([out_paragraph, paragraph_repr], dim=2)
            question_repr = torch.cat([out_question, question_repr], dim=2)
            paragraph_repr = self.p_mlps[i](paragraph_repr)
            question_repr = self.q_mlps[i](question_repr)
            paragraph_repr = apply_rnn(self.paragraph_rnns[i], paragraph_repr)
            question_repr = apply_rnn(self.question_rnns[i], question_repr)

        out = F.relu(self.fc1(paragraph_repr))
        out = self.fc2(out)
        start_scores = out[:, :, 0]
        end_scores = out[:, :, 1]
        return start_scores, end_scores


class JengaDocReader(nn.Module):
    def __init__(self, args, normalize=True, character_dict=None, unk_index=1):

        super(JengaDocReader, self).__init__()
        # Store config
        self.args = args
        self.hidden_size = args.hidden_size
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0,
        )

        if args.learn_unk:
            self.unk_emb = Parameter(torch.zeros(args.embedding_dim))
            # Setting to zero the UNK word embedding
            self.embedding.weight.data[unk_index].zero_()

        if args.learn_hash_emb > 0:
            self.hash_emb = nn.Embedding(args.learn_hash_emb, args.embedding_dim)
            self.hash_emb.weight.data.zero_()


        self.unk_index = unk_index

        # ...maybe keep them fixed
        if args.fix_embeddings:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # ...or maybe register a buffer to fill later for keeping *some* fixed
        if args.tune_partial > 0:
            buffer_size = torch.Size(
                (args.vocab_size - args.tune_partial - 2, args.embedding_dim)
            )
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        if args.rnn_type == 'sru':
            # Import is slow, only do it if necessary
            import sru
            RNN_TYPES['sru'] = sru.SRU

        rnn_type = RNN_TYPES[args.rnn_type]

        # Character-level rnn
        if self.args.character_level:
            assert character_dict is not None
            self.character_embedding = nn.Embedding(
                len(character_dict),
                args.embedding_dim,
                padding_idx=0,
            )
            self.character_rnn = rnn_type(args.embedding_dim, args.character_level_size, num_layers=1, bidirectional=True)

        grid_expander = 1
        if not self.args.grid_horizontal_only:
            grid_expander *= 2
        if not self.args.grid_single_direction:
            grid_expander *= 4
        h_expander = {
            'q_first': 2,
            'c_first': 2,
            'concat': 4,
            'grid': grid_expander,
        }[args.jenga_arch]

        def make_rnn(input_size):
            rnn = rnn_type(input_size, args.hidden_size, num_layers=1, bidirectional=True)
            if self.args.custom_lstm_bias_init:
                for prefix in ('bias_ih_l', 'bias_hh_l'):
                    for i in range(rnn.num_layers):
                        for suffix in ('', '_reverse'):
                            p = getattr(rnn, '{}{}{}'.format(prefix, i, suffix))
                            block_size = p.data.numel() // 4
                            p.data[block_size:2 * block_size] += 1
            return rnn

        n_features = args.num_features if args.jenga_language_features else 0
        if args.jenga_q_c_dot_product:
            n_features += 1
        input_size = 2 * args.embedding_dim + n_features + (4 * args.character_level_size if self.args.character_level else 0)
        if args.jenga_cnn != 'off':
            h_expander = 1
            self.cnns = nn.ModuleList()
            num_filters = args.hidden_size
            if args.jenga_cnn_gating:
                num_filters *= 2
            if self.args.jenga_cnn_start_11:
                self.cnns.append(nn.Conv2d(input_size, num_filters, [1, 1]))
            for i in range(args.blocks):
                width = args.jenga_kernel_width
                dilation = 1
                padding = (dilation * (width - 1)) // 2  # assuming stride = 1
                current_layer_input_size = input_size if i == 0 and not self.args.jenga_cnn_start_11 else args.hidden_size
                if args.jenga_cnn == '2d':
                    self.cnns.append(nn.Conv2d(current_layer_input_size, num_filters, [width, width], padding=(padding,padding)))
                else:
                    self.cnns.append(nn.Conv2d(current_layer_input_size, num_filters, [1, width], padding=(0, padding)))
                    self.cnns.append(nn.Conv2d(args.hidden_size, num_filters, [width, 1], padding=(padding, 0)))
        if args.jenga_self_attention and args.jenga_attention_cat:
            h_expander = h_expander * 2
        if args.jenga_arch == 'grid':
            self.grid_rnns = nn.ModuleList()
            for i in range(args.blocks):
                self.grid_rnns.append(layers.GridLSTM(
                    input_size if i == 0 else h_expander * args.hidden_size,
                    args.hidden_size,
                    self.args.grid_single_direction,
                    self.args.grid_horizontal_only))
        else:
            self.odd_rnns = nn.ModuleList()
            self.even_rnns = nn.ModuleList()
            for i in range(args.blocks):
                self.odd_rnns.append(make_rnn(input_size
                    if i == 0
                    else h_expander * args.hidden_size))
                self.even_rnns.append(make_rnn(input_size
                    if i == 0 and args.jenga_arch == 'concat'
                    else h_expander * args.hidden_size))

        self.last_rnn = make_rnn(h_expander * args.hidden_size * (args.blocks if args.concat_rnn_layers else 1))

        self.init_last_layer()

        if args.jenga_word_dropout > 0:
            self.word_dropout = torch.nn.Dropout2d(p=args.jenga_word_dropout)
        if self.args.jenga_language_features and args.jenga_lf_dropout > 0:
            self.lf_dropout = torch.nn.Dropout2d(p=args.jenga_lf_dropout)

        if self.args.use_word_dropout_all_layers:
            self.mid_layers_dropout = torch.nn.Dropout2d(p=args.dropout_jenga)
        else:
            self.mid_layers_dropout = torch.nn.Dropout(p=args.dropout_jenga)

        if self.args.jenga_self_attention:
            nb_attention_heads = 2 * args.blocks * args.jenga_nb_attention_head
            proj_size = 2 * args.hidden_size // args.jenga_nb_attention_head
            self.self_attn = nn.ModuleList([layers.SeqSelfAttnMatch(2 * args.hidden_size, identity=False, proj_size=proj_size, extra_scaling=args.jenga_attention_extra_scaling, remove_id=args.jenga_attention_cat)
                                            for i in range(nb_attention_heads)])

        if self.args.jenga_flatten_question:
            self.flat_proj = nn.Linear(h_expander * args.hidden_size, h_expander * args.hidden_size)

        if self.args.jenga_use_question_final:
            self.question_rnn = nn.LSTM(args.embedding_dim, args.hidden_size, num_layers=2, bidirectional=True)
            self.question_proj = nn.Linear(2 * args.hidden_size, 1)
            if self.args.jenga_iterative_decode:
                self.decode_rnn = nn.GRU(2 * args.hidden_size, 2 * args.hidden_size, num_layers=1)
                self.iterative_decode_lin5 = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)  # corresponds to w5 matrix in SAN paper


    def init_last_layer(self):
        args = self.args
        n_last_hidden_dim = args.hidden_size
        if args.jenga_cnn == 'off':
            n_last_hidden_dim *= 2
        if args.jenga_shortcut_embedding:
            n_last_hidden_dim += args.embedding_dim
        if args.jenga_use_start_for_end:
            self.fc_start = nn.Linear(n_last_hidden_dim, 1)
            self.fc_end_bil = nn.Bilinear(n_last_hidden_dim, n_last_hidden_dim, 1)
            self.fc_end_lin = nn.Linear(n_last_hidden_dim, 1)
        elif args.jenga_use_question_final:
            self.fc = nn.Linear(n_last_hidden_dim, 2 * n_last_hidden_dim)
        elif args.span_joint_optimization:
            if self.args.length_in_jo:
                n_classifier_features = 2 * n_last_hidden_dim + 1
            else:
                n_classifier_features = 2 * n_last_hidden_dim
            if self.args.jo_bilinear:
                self.bilin = nn.Bilinear(n_last_hidden_dim, n_classifier_features - n_last_hidden_dim, 1)
            elif self.args.jo_projection_hidden_size is not None:
                self.jo_projection = nn.Linear(n_classifier_features, self.args.jo_projection_hidden_size)
                self.fc = nn.Linear(self.args.jo_projection_hidden_size, 1)
            else:
                self.fc = nn.Linear(n_classifier_features, 1)
        else:
            self.fc = nn.Linear(n_last_hidden_dim, 2)


    def forward(self, ex):
        """
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1 = ex['x1']
        x1_f = ex['x1_f']
        x1_mask = ex['x1_mask']
        x2 = ex['x2']
        x2_mask = ex['x2_mask']
        # Embed both document and question
        x1_emb = self.embedding(x1)  # [batch, len_d, emb_size]
        x2_emb = self.embedding(x2)  # [batch, len_q, emb_size]

        if hasattr(self.args, 'learn_unk') and self.args.learn_unk:
            # Add the learned 'UNK' embedding
            unk_emb = self.unk_emb.unsqueeze(0).unsqueeze(0)
            x1_emb += (x1 == self.unk_index).float().unsqueeze(2).expand_as(x1_emb) * unk_emb.expand_as(x1_emb)
            x2_emb += (x2 == self.unk_index).float().unsqueeze(2).expand_as(x2_emb) * unk_emb.expand_as(x2_emb)

        if hasattr(self.args, 'learn_hash_emb') and self.args.learn_hash_emb > 0:
            x1_emb += self.hash_emb(x1 % self.args.learn_hash_emb)
            x2_emb += self.hash_emb(x2 % self.args.learn_hash_emb)

        if self.args.character_level:
            x1_char = ex['x1_char']
            x1_char_lengths = ex['x1_char_lengths']
            x2_char = ex['x2_char']
            x2_char_lengths = ex['x2_char_lengths']
            def embed_chars(char, char_lengths, mask):
                batch_size = char.size(0)
                seq_lengths = (mask == 0).long().sum(dim=1).data.cpu()
                num_tokens = seq_lengths.sum()
                # Build 2d tensor with all characters of all tokens of all examples, one token per row
                char_padded = char.data.new(num_tokens, char_lengths.data.max()).zero_()
                # Build 1d tensor with all lengths of all tokens of all examples
                char_lengths_unpadded = char_lengths.data.new(num_tokens).zero_()
                dst_char_length_offset = 0
                dst_char_idx = 0
                for i, l1 in enumerate(seq_lengths):
                    src_char_offset = 0
                    lengths = char_lengths.data[i, :l1]
                    char_lengths_unpadded[dst_char_length_offset:dst_char_length_offset + l1].copy_(lengths)
                    dst_char_length_offset += l1
                    for l2 in lengths:
                        char_padded[dst_char_idx, :l2].copy_(char.data[i, src_char_offset:src_char_offset + l2])
                        src_char_offset += l2
                        dst_char_idx += 1
                # Sort token lengths in descending order for packing. Also compute reverse order
                sorted_char_lengths, sorted_char_order = char_lengths_unpadded.sort(0, True)
                reverse_char_order = sorted_char_order.new(sorted_char_order.numel())
                for i, l in enumerate(sorted_char_order):
                    reverse_char_order[l] = i
                # Create variables and send tensors to GPU if necessary
                def make_var(x):
                    return Variable(x.cuda(async=True) if self.args.cuda else x)
                char_padded, sorted_char_lengths, sorted_char_order, reverse_char_order = \
                    map(make_var, (char_padded, sorted_char_lengths, sorted_char_order, reverse_char_order))
                # Sort char_padded by decreasing token length
                sorted_char_padded = char_padded.index_select(0, sorted_char_order)
                # Lookup character embeddings
                sorted_emb_padded = self.character_embedding(sorted_char_padded)
                # Pack input for RNN
                packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_emb_padded, sorted_char_lengths.data.tolist(), batch_first=True)
                # Feed to RNN and max-pool to get embedding for each token
                rnn_output = torch.nn.utils.rnn.pad_packed_sequence(self.character_rnn(packed)[0], batch_first=True)[0]
                # Max-pool to get fixed-size representation for each token
                rnn_output[rnn_output == 0] = -float('inf')  # Make sure not to include padding in the max-pool
                sorted_token_emb = rnn_output.max(1)[0]
                # Recover original token order
                token_emb = sorted_token_emb.index_select(0, reverse_char_order)
                # Sequence-wise padding for token embeddings
                padded_token_emb = Variable(token_emb.data.new(batch_size, mask.size(1), token_emb.size(1)).zero_())
                offset = 0
                for i, l in enumerate(seq_lengths):
                    padded_token_emb[i, :l, :] = token_emb[offset:offset + l]
                    offset += l
                return padded_token_emb
            x1_char_emb = embed_chars(x1_char, x1_char_lengths, x1_mask)
            x2_char_emb = embed_chars(x2_char, x2_char_lengths, x2_mask)
            x1_emb = torch.cat([x1_emb, x1_char_emb], dim=2)
            x2_emb = torch.cat([x2_emb, x2_char_emb], dim=2)

        batch_size, c_size, emb_size = x1_emb.size()
        q_size = x2.size(1)

        if self.args.jenga_word_dropout > 0 and self.args.jenga_fix_word_dropout:
            x1_emb = self.word_dropout(x1_emb)
            x2_emb = self.word_dropout(x2_emb)

        x2_emb = x2_emb.transpose(1, 2).unsqueeze(2) # [batch, emb_size, 1, len_q]

        if self.args.jenga_q_c_dot_product:
            q_c_dot_product = x1_emb.view(batch_size, c_size, emb_size).bmm(x2_emb.squeeze(2))
            # [batch, c_size, q_size]

        x1_emb = x1_emb.transpose(1, 2).unsqueeze(3) # [batch, emb_size, len_d, 1]

        # Dropout on embeddings
        if self.args.jenga_word_dropout > 0:
            if not self.args.jenga_fix_word_dropout:
                x1_emb = self.word_dropout(x1_emb)
                x2_emb = self.word_dropout(x2_emb)
        elif self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(
                x1_emb, p=self.args.dropout_emb, training=self.training
            )
            x2_emb = nn.functional.dropout(
                x2_emb, p=self.args.dropout_emb, training=self.training
            )

        if self.args.jenga_language_features and self.args.jenga_lf_dropout > 0:
            x1_f = self.lf_dropout(x1_f)

        # Build "image"

        x_size = [batch_size, emb_size, c_size, q_size]

        expanded_context = x1_emb.expand(*x_size)
        expanded_question = x2_emb.expand(*x_size)
        if self.args.jenga_emb_diff_mode > 0:
            expanded_context = expanded_context - expanded_question
            if self.args.jenga_emb_diff_mode == 2:
                expanded_context.data.abs_()
            elif self.args.jenga_emb_diff_mode == 3:
                expanded_context.data.pow_(2)
        if self.args.jenga_emb_diff_mode < 0:
            expanded_question = expanded_question - expanded_context
            if self.args.jenga_emb_diff_mode == -2:
                expanded_question.data.abs_()
            elif self.args.jenga_emb_diff_mode == -3:
                expanded_question.data.pow_(2)

        tensors = [ expanded_context, expanded_question ]
        if self.args.jenga_language_features:
            tensors.append(x1_f.transpose(1, 2).unsqueeze(3).expand(batch_size, self.args.num_features, c_size, q_size))
        if self.args.jenga_q_c_dot_product:
            tensors.append(q_c_dot_product.view(batch_size, 1, c_size, q_size))

        x_img = torch.cat(tensors, dim=1) # [batch * (2 * emb_size + nfeat) * len_d * len_q]

        if self.args.jenga_self_attention:
            doc_mask = x1_mask.unsqueeze(1).unsqueeze(3).expand(batch_size, 1, c_size, q_size)
            q_mask = x2_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, c_size, q_size)
            x_mask = doc_mask + q_mask - doc_mask * q_mask
            x_mask_even = x_mask.transpose(1, 2).contiguous().view(batch_size * c_size, q_size)
            x_mask_odd = x_mask.transpose(1, 3).contiguous().view(batch_size * q_size, c_size)

        if self.args.add_context_maxpool:
            transposed_context_mask = x1_mask.transpose(0, 1).contiguous().unsqueeze(2).expand(c_size, batch_size, q_size).contiguous()

        def to_context_direction(x_img):
            # Put in context direction: [len_d, batch * len_q, input_size]
            #    +-----------------+
            #    |---------------->|
            #    |---------------->|
            #    |---------------->|
            #    +-----------------+
            return x_img.permute(2, 0, 3, 1).contiguous().view(c_size, batch_size * q_size, -1), True

        def to_question_direction(x_img):
            # Put in question direction: [len_q, batch * len_d, input_size]
            #    +-----------------+
            #    | ^ ^ ^ ^ ^ ^ ^ ^ |
            #    | | | | | | | | | |
            #    | | | | | | | | | |
            #    +-----------------+
            return x_img.permute(3, 0, 2, 1).contiguous().view(q_size, batch_size * c_size, -1), False

        def swap_directions(x_img, is_context_direction):
            # Toggle between question and context directions
            #    +-----------------+      +-----------------+
            #    |---------------->|      | ^ ^ ^ ^ ^ ^ ^ ^ |
            #    |---------------->|  =>  | | | | | | | | | |
            #    |---------------->|      | | | | | | | | | |
            #    +-----------------+      +-----------------+
            #
            #    +-----------------+      +-----------------+
            #    | ^ ^ ^ ^ ^ ^ ^ ^ |      |---------------->|
            #    | | | | | | | | | |  =>  |---------------->|
            #    | | | | | | | | | |      |---------------->|
            #    +-----------------+      +-----------------+
            size1, size2 = x_img.size(0), x_img.size(1) // batch_size
            return x_img.view(size1, batch_size, size2, -1) \
                .transpose(0, 2).contiguous() \
                .view(size2, batch_size * size1, -1), not is_context_direction

        def apply_rnn(rnn, x_img, is_context_direction):
            out = rnn(x_img)
            if isinstance(out, tuple):
                out = out[0]

            if is_context_direction and self.args.add_context_maxpool:
                # Add maxpool over whole context to each element
                # Cheap & easy way to propagate paragraph-level info
                expanded_mask = transposed_context_mask.view(c_size, batch_size * q_size, 1).expand_as(out)
                out = out.masked_fill(expanded_mask, -float('inf'))
                out += out.max(0)[0].unsqueeze(0).expand_as(out)
                out = out.masked_fill(expanded_mask, 0)

            if self.args.jenga_do_b4_res:
                out = self.mid_layers_dropout(out)
            if self.args.jenga_residual_rnns and x_img.size() == out.size():
                if self.args.jenga_fix_residuals:
                    out = (out + x_img) * math.sqrt(0.5)
                else:
                    out = out + x_img
            return out

        if self.args.jenga_cnn != 'off':
            layer_outputs = []
            if self.args.jenga_cnn == '1d_context_only':
                # x_img = x_img.permute(3, 0, 2, 1) # [len_q, batch, len_d, h]
                x_img = x_img.permute(2, 3, 0, 1) # [c_size, q_size, batch_size, input_size]
                for grid_rnn in self.grid_rnns:
                    x_img = grid_rnn(x_img)
                    x_img = self.mid_layers_dropout(x_img)
                    layer_outputs.append(x_img)
                if self.args.concat_rnn_layers:
                    x_img = torch.cat(layer_outputs, dim=3)
                # x_img = x_img.permute(1, 3, 2, 0) # [batch, h, len_d, len_q]
                x_img = x_img.permute(2, 3, 0, 1)
            else:
                # [batch, h, len_d, len_q]
                for cnn in self.cnns:
                    x_img = cnn(x_img)
                    if self.args.jenga_cnn_gating:
                        x_img = F.glu(x_img, dim=1)
                    else:
                        x_img = F.relu(x_img)
                    x_img = self.mid_layers_dropout(x_img)
                    layer_outputs.append(x_img)
                if self.args.concat_rnn_layers:
                    x_img = torch.cat(layer_outputs, dim=3)
            x_img, is_context_direction = to_question_direction(x_img)
        elif self.args.jenga_arch == 'grid':
            layer_outputs = []
            # x_img = x_img.permute(3, 0, 2, 1) # [len_q, batch, len_d, h]
            x_img = x_img.permute(2, 3, 0, 1) # [c_size, q_size, batch_size, channels]
            for grid_rnn in self.grid_rnns:
                x_img = grid_rnn(x_img)
                x_img = self.mid_layers_dropout(x_img)
                layer_outputs.append(x_img)
            if self.args.concat_rnn_layers:
                x_img = torch.cat(layer_outputs, dim=3)
            # x_img = x_img.permute(1, 3, 2, 0) # [batch, h, len_d, len_q]
            x_img = x_img.permute(2, 3, 0, 1) # [batch, channels, c_size, q_size]
            x_img, is_context_direction = to_question_direction(x_img)
        else:
            if self.args.jenga_arch == 'c_first':
                x_img, is_context_direction = to_context_direction(x_img)
            else:  # jenga_arch == 'q_first' or 'concat'
                x_img, is_context_direction = to_question_direction(x_img)

            layer_outputs = []
            if self.args.jenga_arch == 'concat':
                # Apply RNN in both directions every time and concatenate
                #                     ^
                #                     |
                #          +----------+-----------+
                #          |                      |
                # +-----------------+    +-----------------+
                # |---------------->|    | ^ ^ ^ ^ ^ ^ ^ ^ |
                # |---------------->|    | | | | | | | | | |
                # |---------------->|    | | | | | | | | | |
                # +-----------------+    +-----------------+
                #          ^                      ^
                #          |                      |
                #          +-----------+----------+
                #                      |
                for odd_rnn, even_rnn in zip(self.odd_rnns, self.even_rnns):
                    odd_x_img = apply_rnn(odd_rnn, x_img, is_context_direction)
                    swapped_x_img, is_context_direction = swap_directions(x_img, is_context_direction)
                    even_x_img = apply_rnn(even_rnn, swapped_x_img, is_context_direction)
                    swapped_even_x_img, is_context_direction = swap_directions(even_x_img, is_context_direction)
                    x_img = torch.cat([odd_x_img, swapped_even_x_img], dim=2)
                    x_img = self.mid_layers_dropout(x_img)
                    layer_outputs.append(x_img)
            else:
                # Alternate RNN directions
                #                     ^
                #                     |
                #            +-----------------+
                #            | ^ ^ ^ ^ ^ ^ ^ ^ |
                #            | | | | | | | | | |
                #            | | | | | | | | | |
                #            +-----------------+
                #                     ^
                #                     |
                #            +-----------------+
                #            |---------------->|
                #            |---------------->|
                #            |---------------->|
                #            +-----------------+
                #                     ^
                #                     |

                def self_attention(x_img, x_mask, i):
                    if self.args.jenga_self_attention:
                        x_img = x_img.view(x_img.size(0) * batch_size, x_img.size(1) // batch_size, -1)
                        if self.args.jenga_nb_attention_head == 1:
                            x_img = self.self_attn[i](x_img, x_mask,
                                                      scale_dot_product=self.args.jenga_scale_attention_output,
                                                      return_projected=self.args.jenga_attention_projected,
                                                      residual_attention=self.args.jenga_attention_residuals)
                            i += 1
                        else:
                            attn = []
                            for j in range(self.args.jenga_nb_attention_head):
                                attn.append(self.self_attn[i](x_img, x_mask,
                                                          scale_dot_product=self.args.jenga_scale_attention_output,
                                                          return_projected=True,
                                                          residual_attention=self.args.jenga_attention_residuals))
                                i += 1
                            x_img = torch.cat(attn, dim=2)
                        x_img = x_img.view(x_img.size(0) // batch_size, x_img.size(1) * batch_size, -1)
                    return x_img, i
                self_attention_idx = 0
                for i, (odd_rnn, even_rnn) in enumerate(zip(self.odd_rnns, self.even_rnns)):
                    if not self.args.jenga_flatten_question:
                        x_img = apply_rnn(odd_rnn, x_img, is_context_direction)
                        if not self.args.jenga_do_b4_res:
                            x_img = self.mid_layers_dropout(x_img)
                    elif i == 0:
                        x_img = apply_rnn(odd_rnn, x_img, is_context_direction)
                        if not self.args.jenga_do_b4_res:
                            x_img = self.mid_layers_dropout(x_img)
                        x_img = x_img.max(0)[0].unsqueeze(0)

                    if self.args.jenga_self_attention:
                        x_img, self_attention_idx = self_attention(x_img, x_mask_odd, self_attention_idx)

                    x_img, is_context_direction = swap_directions(x_img, is_context_direction)
                    x_img = apply_rnn(even_rnn, x_img, is_context_direction)
                    if not self.args.jenga_do_b4_res:
                        x_img = self.mid_layers_dropout(x_img)
                    if self.args.jenga_self_attention:
                        x_img, self_attention_idx = self_attention(x_img, x_mask_even, self_attention_idx)
                    x_img, is_context_direction = swap_directions(x_img, is_context_direction)

                    layer_outputs.append(x_img)
            if self.args.concat_rnn_layers:
                x_img = torch.cat(layer_outputs, dim=2)
            if self.args.jenga_arch == 'c_first':
                # Put back in question direction before applying final RNN
                x_img, is_context_direction = swap_directions(x_img, is_context_direction)
        if self.args.jenga_cnn == 'off':
            out = apply_rnn(self.last_rnn, x_img, is_context_direction) # [h, batch * w, 2 * channels]
            if not self.args.jenga_do_b4_res and not self.args.jenga_remove_last_do:
                out = self.mid_layers_dropout(out)
        else:
            out = x_img
        return self.final_layer(out, batch_size, c_size, x1_mask, x2_emb)

    def final_layer(self, out, batch_size, c_size, x1_mask, x2_emb):
        if self.args.pool_type == 'avg':
            out = out.mean(0)
        else:
            # assuming max pooling
            out = out.max(0)[0] # [batch * c_size, 2 * channels]

        if self.args.jenga_shortcut_embedding:
            #Reminder: x1_emb is [batch, emb_size, len_d, 1]
            doc_emb = x1_emb.transpose(1, 2).contiguous().view(batch_size * c_size, -1)
            out = torch.cat([out, doc_emb], dim=1)


        if self.args.span_joint_optimization:
            hidden_size = self.args.hidden_size
            max_len = self.args.max_len

            out = out.view(batch_size, c_size, 2 * hidden_size)
            sliding = self.slide_tensor(out, None, max_len)# batch_size, max_len, c_size, 2 * hidden_size

            fixed_features = out.unsqueeze(1).expand(batch_size, max_len, c_size, 2 * hidden_size)

            if self.args.length_in_jo:
                n_total_features = 4 * self.args.hidden_size + 1
                len_range = torch.arange(0, max_len)
                if self.args.cuda:
                    len_range = len_range.cuda(async=True)
                len_index = Variable(len_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batch_size, max_len, c_size, 1), requires_grad=False)
                big_table = torch.cat([sliding, fixed_features, len_index], dim=3) # batch_size, max_len, c_size, 4 * hidden_size + 1
            else:
                n_total_features = 4 * self.args.hidden_size
                big_table = torch.cat([sliding, fixed_features], dim=3) # batch_size, max_len, c_size, 4 * hidden_size

            se_tensor = big_table.view(batch_size, self.args.max_len * c_size, n_total_features) # batch x (max_len * c_size) x 4*channels

            if self.args.jo_bilinear:
                se_tensor = se_tensor.view(-1, n_total_features)
                joint_score = self.bilin(se_tensor.narrow(1, 0, 2 * self.args.hidden_size),
                                         se_tensor.narrow(1, 2 * self.args.hidden_size, n_total_features - 2 * self.args.hidden_size))
                joint_score = joint_score.view(batch_size, self.args.max_len * c_size)
            else:
                if self.args.jo_projection_hidden_size is not None:
                    se_tensor = F.relu(self.jo_projection(se_tensor))

                joint_score = self.fc(se_tensor).squeeze(dim=2)
                # [batch, max_len * c_size]

            mask = self.slide_tensor(x1_mask.unsqueeze(2), 1, max_len).view(batch_size, max_len * c_size)

            joint_score.data.masked_fill_(mask.data, -float('inf'))

            if self.training:
                joint_score = F.log_softmax(joint_score)
            else:
                joint_score = F.softmax(joint_score)
            return joint_score # [batch, max_len * c_size]

        if self.args.jenga_use_start_for_end:
            start_scores = self.fc_start(out).view(batch_size, c_size)
            normalized_scores = F.softmax(start_scores, dim=1).unsqueeze(1)  # .detach()  # [b, 1, c]
            start_repr = normalized_scores.bmm(out.view(batch_size, c_size, -1)) # [b, 1, 2h]
            start_repr = F.normalize(start_repr, dim=2)
            start_repr = start_repr.expand(batch_size, c_size, 2 * self.args.hidden_size)
            end_scores = (self.fc_end_lin(out) + self.fc_end_bil(out, start_repr.contiguous().view_as(out))).view(batch_size, c_size)
            # end_scores = self.fc_end(out).view(batch_size, c_size)
            return start_scores, end_scores
        elif self.args.jenga_use_question_final:
            # x2_emb is [batch, emb_size, 1, len_q]
            x2_emb = x2_emb.squeeze(2).permute(2, 0, 1).contiguous()
            # now x2_emb is [len_q, batch, emb_size]
            encoded_question = self.question_rnn(x2_emb)[0].transpose(0, 1) # [batch, len_q, 2h]
            encoded_question = self.mid_layers_dropout(encoded_question)
            weights = self.question_proj(encoded_question).squeeze(2) # [batch, len_q]
            weights = F.softmax(weights, dim=1).unsqueeze(1) # [b, 1, len_q]

            question_repr = weights.bmm(encoded_question).squeeze(1) # [batch, 2h]
            # question_repr is [batch, hidden_size * 2]
            if self.args.jenga_iterative_decode:
                encoded_context = out.view(batch_size, c_size, -1) # [b, c, 2h]
                def compute_x(s_t):
                    # s_t is [1, b, 2h]
                    projected = self.iterative_decode_lin5(s_t.view(batch_size, -1)).unsqueeze(2)
                    weights = encoded_context.bmm(projected).view(batch_size, c_size)
                    # weights is [batch, len_c]
                    weights = F.softmax(weights, dim=1).unsqueeze(1)
                    x_t = weights.bmm(encoded_context)
                    return x_t.view(1, batch_size, -1)
                current_s = question_repr.view(1, batch_size, 2 * self.args.hidden_size)
                all_s = []
                n_steps = 5
                for i in range(n_steps):
                    current_x = compute_x(current_s)
                    _, current_s = self.decode_rnn(current_x, current_s)
                    # current_s: [1, batch, 2 * h]
                    all_s.append(current_s.squeeze(0))
                all_s = torch.stack(all_s, dim=1)
                # [batch, n_steps, 2h]
            else:
                all_s = question_repr.unsqueeze(1)
                n_steps = 1
            projected_s = self.fc(all_s).transpose(1, 2).contiguous()  # [batch, hidden_size * 4, n_steps]
            projected_s = projected_s.view(batch_size, 2 * self.args.hidden_size, 2 * n_steps)
            out = out.view(batch_size, c_size, 2 * self.args.hidden_size)
            out = out.bmm(projected_s).view(batch_size, c_size, 2, n_steps)
            start_scores = out[:, :, 0, :]
            end_scores = out[:, :, 1, :]
            return start_scores, end_scores
        else:
            out = self.fc(out) # [batch * w, 2]
        out = out.view(batch_size, c_size, 2) # [batch, w, 2]

        start_scores = out[:, :, 0]
        end_scores = out[:, :, 1]
        return start_scores, end_scores

    def slide_tensor(self, data, filler_value, max_len):
        batch_size, c_size, n_features = data.size()
        n_patches = (max_len // c_size) + 1 # most of the time this will just be 1
        sliding = data.unsqueeze(1).expand(batch_size, max_len + n_patches, c_size, n_features).contiguous()
        sliding = sliding.view(batch_size, (max_len + n_patches) * c_size, n_features)
        sliding = sliding.narrow(1, 0, max_len * (c_size + 1)).contiguous().view(batch_size, max_len, (c_size + 1), n_features)
        sliding = sliding.narrow(2, 0, c_size).contiguous().view(batch_size, max_len, c_size, n_features)
        if filler_value is not None:
            cell_not_in_table = (torch.arange(0, max_len).unsqueeze(1).expand(max_len, c_size) + torch.arange(0, c_size).unsqueeze(0).expand(max_len, c_size)) >= c_size
            if self.args.cuda:
                cell_not_in_table = cell_not_in_table.cuda(async=True)
            cell_not_in_table = Variable(cell_not_in_table, requires_grad = False)
            sliding = sliding.masked_fill(cell_not_in_table.unsqueeze(0).unsqueeze(3).expand_as(sliding), filler_value)
        return sliding

class DocReader(nn.Module):
    def __init__(self, args, normalize=True):

        super(DocReader, self).__init__()
        # Store config
        self.args = args
        self.hidden_size = args.hidden_size
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0
        )

        # ...maybe keep them fixed
        if args.fix_embeddings:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # ...or maybe register a buffer to fill later for keeping *some* fixed
        if args.tune_partial > 0:
            buffer_size = torch.Size(
                (args.vocab_size - args.tune_partial - 2, args.embedding_dim)
            )
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        self.doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb and not args.use_gated_question_attention_input:
            self.doc_input_size += args.embedding_dim
        if args.paragraph_self_attn:
            self.doc_input_size += args.embedding_dim
        if args.qemb_match_first_layer:
            self.doc_input_size += args.embedding_dim
        if args.use_gated_question_attention_input:
            self.input_gates = nn.Linear(args.embedding_dim, args.embedding_dim)
        # size of output of question encoder.
        self.question_hidden_size = None
        if args.model == 'cnn_rnn' or args.model == 'cnn':
            self.question_hidden_size = sum(args.num_filters)
            if args.concat_rnn_layers:
                self.question_hidden_size *= args.question_layers

        self.pool_context_output_layer = None

    def forward(self, ex):
        """
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1 = ex['x1']
        x1_f = ex['x1_f']
        x1_mask = ex['x1_mask']
        x2 = ex['x2']
        x2_mask = ex['x2_mask']
        # Embed both document and question
        x1_emb = self.embedding(x1) # [batch * len_d * emb_size]
        x2_emb = self.embedding(x2) # [batch * len_q * emb_size]

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(
                x1_emb, p=self.args.dropout_emb, training=self.training
            )
            x2_emb = nn.functional.dropout(
                x2_emb, p=self.args.dropout_emb, training=self.training
            )

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            if self.args.use_gated_question_attention_input:
                x2_weighted_emb = self.input_gates(
                    x2_weighted_emb.view(-1, x2_weighted_emb.size(2))
                ).view(x2_weighted_emb.size())
            drnn_input = torch.cat([x1_emb, x2_weighted_emb], 2)

        if self.args.paragraph_self_attn:
            self_weighted_emb = self.qemb_match(x1_emb, x1_emb, x1_mask)
            drnn_input = torch.cat([drnn_input, self_weighted_emb], 2)

        if self.args.use_gated_question_attention_input:
            drnn_input = F.glu(drnn_input, dim=2)

        drnn_input = torch.cat([drnn_input, x1_f], 2) # add the manual features

        question_hiddens, question_hidden = self.encode_question(
            x2_emb, x2_mask
        )
        if self.args.scale_gradients:
            question_hiddens = grad_multiply(question_hiddens, 1.0/self.args.cnn_doc_layers)

        doc_hiddens = self.encode_doc(
            drnn_input, x1_mask, question_hiddens, x2_mask, x2_emb
        )

        # Predict start and end positions
        if self.pool_context_output_layer:
            if self.pool_type == 'concat':
                doc_hiddens = F.conv1d(
                    doc_hiddens.transpose(1, 2),
                    self.concat_filters,
                    padding=self.padding
                )
                doc_hiddens = doc_hiddens.transpose(1, 2)
            else:
                doc_hiddens = self.pool_net(doc_hiddens)
            if self.dropout_pooled_context > 0.0:
                F.dropout(
                    doc_hiddens,
                    p=self.dropout_pooled_context,
                    training=self.training
                )
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)

        return start_scores, end_scores

    def encode_doc(self, token_embeddings, token_mask):
        """
        Abstract method for encoding docs
        """
        raise NotImplementedError('The specific encoder will implement this')

    def encode_question(self, question_embeddings, question_token_mask):
        """
        Abstract method for encoding questions
        """
        raise NotImplementedError('The specific encoder will implement this')


class CNNDocRNNQuestionReader(DocReader):
    def __init__(self, args, normalize=True):

        super(CNNDocRNNQuestionReader, self).__init__(args, normalize=True)

        # RNN question encoder
        # dividing the hidden size by 2 because it gets doubled because of BiLSTMs
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=sum(args.num_filters) // 2,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        self.doc_cnn = layers.StackedCNN(
            input_size=self.doc_input_size,
            num_filters=args.num_filters,
            num_layers=args.cnn_doc_layers,
            kernel_widths=args.kernel_widths,
            concat_layers=args.concat_cnn_layers,
            dropout_rate=args.dropout_cnn,
            dropout_output=args.dropout_cnn_output,
            use_dilation=args.use_dilation,
            use_gated_cnns=args.use_gated_cnns,
            use_residual_layer=args.use_residual_layer,
            use_word_dropout=args.dropout_words,
            use_gated_question_attention=args.use_gated_question_attention,
            question_hidden_size=self.question_hidden_size,
            attn_type=args.attn_type,
            share_gates_param=args.share_gates_param,
            use_dropout_2d_filter=args.use_dropout_2d_filter,
            use_word_dropout_all_layers=args.use_word_dropout_all_layers,
            qemb_match_all_layers=args.qemb_match_all_layers,
            qemb_match_last_layer=args.qemb_match_last_layer,
            qemb_match_first_layer=args.qemb_match_first_layer,
            qemb_project_before_softmax=args.qemb_project_before_softmax
        )
        # Output sizes of rnn encoders
        doc_hidden_size = sum(args.num_filters)
        question_hidden_size = sum(args.num_filters)
        if args.concat_cnn_layers:
            doc_hidden_size *= args.cnn_doc_layers
        if args.qemb_match_last_layer and not args.use_gated_question_attention:
            doc_hidden_size += sum(args.num_filters)
        if args.concat_rnn_layers:
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # pooling at the output layer
        self.pool_context_output_layer = args.pool_context_output_layer
        self.dropout_pooled_context = args.dropout_pooled_context
        self.pool_type = args.pool_type if self.pool_context_output_layer else None
        self.concat_filters = None
        self.padding = None
        if self.pool_context_output_layer:
            assert args.context_size % 2 == 1  # assumming stride = 1.
            self.padding = (args.context_size - 1) // 2
            if args.pool_type == 'avg':
                self.pool_net = nn.AvgPool1d(
                    args.context_size, stride=1, padding=self.padding
                )
            elif args.pool_type == 'max':
                self.pool_net = nn.MaxPool1d(
                    args.context_size, stride=1, padding=self.padding
                )
            elif args.pool_type == 'concat':
                # create a filter for concatenating input paragraph vectors
                zeros = torch.zeros(
                    args.context_size * doc_hidden_size, doc_hidden_size,
                    args.context_size
                )
                if args.cuda:
                    zeros = zeros.cuda(async=True)

                self.concat_filters = Variable(zeros, requires_grad=False)
                pairs = [
                    (i, j)
                    for j in range(args.context_size)
                    for i in range(doc_hidden_size)
                ]
                for i, p in enumerate(pairs):
                    self.concat_filters[i, p[0], p[1]
                                       ] = 1  # this is the conv filter
                doc_hidden_size *= args.context_size
            else:
                raise NotImplementedError(
                    'currently only max and avg. pooling are supported'
                )

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def encode_doc(
        self, token_embeddings, token_mask, question_embeddings, question_mask, question_emb
    ):
        """
        Encode paragraph with a multilayer CNN.
        token_embeddings: (B, seq_length, hidden_size)
        token_mask: (B, seq_length)
        return: per token representation calculated by CNN (B, seq_length, num_filters).
        """
        out = self.doc_cnn(
            token_embeddings, token_mask, question_embeddings, question_mask, question_emb
        )
        return out

    def encode_question(self, question_embeddings, question_token_mask):
        """
        Encodes the question string. Since unlike doc encoder, we need one vector
        for a question, we need to combine the token representations. We currently
        support avging and self attention (Sec 3.2, Question encoding)
        question_embeddings: (B, seq_length, hidden_size)
        question_token_mask: (B, seq_length)
        """
        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(
            question_embeddings, question_token_mask
        )
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(
                question_hiddens, question_token_mask
            )
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(
                question_hiddens, question_token_mask
            )
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # [B, seq_length, 2*hidden_size], [B, 2*hidden_size]
        return question_hiddens, question_hidden


class CNNDocReader(DocReader):
    def __init__(self, args, normalize=True):

        super(CNNDocReader, self).__init__(args, normalize=True)

        # CNN doc encoder
        self.doc_cnn = layers.StackedCNN(
            input_size=self.doc_input_size,
            num_filters=args.num_filters,
            kernel_widths=args.kernel_widths,
            num_layers=args.cnn_doc_layers,
            concat_layers=args.concat_cnn_layers,
            dropout_rate=args.dropout_cnn,
            dropout_output=args.dropout_cnn_output,
            use_dilation=args.use_dilation,
            use_gated_cnns=args.use_gated_cnns,
            use_residual_layer=args.use_residual_layer,
            use_word_dropout=args.dropout_words,
            use_dropout_2d_filter=args.dropout_2d_filter,
            use_gated_question_attention=args.use_gated_question_attention
        )

        # CNN question encoder
        self.question_cnn = layers.StackedCNN(
            input_size=args.embedding_dim,
            num_filters=args.num_filters,
            kernel_widths=args.kernel_widths,
            num_layers=args.cnn_question_layers,
            concat_layers=args.concat_cnn_layers,
            dropout_rate=args.dropout_cnn,
            dropout_output=args.dropout_cnn_output,
            use_dilation=args.use_dilation,
            use_gated_cnns=args.use_gated_cnns,
            use_residual_layer=args.use_residual_layer,
            use_word_dropout=args.dropout_words,
            use_dropout_2d_filter=args.dropout_2d_filter,
            use_gated_question_attention=args.use_gated_question_attention
        )

        # Output sizes of rnn encoders
        doc_hidden_size = sum(args.num_filters)
        question_hidden_size = sum(args.num_filters)
        if args.concat_cnn_layers:
            doc_hidden_size *= args.cnn_doc_layers
            question_hidden_size *= args.cnn_question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn', 'max_pool']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        if args.question_merge == 'max_pool':
            self.self_attn = layers.MaxPoolOverTime()

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def encode_doc(self, token_embeddings, token_mask):
        """
        Encode paragraph with a multilayer CNN.
        token_embeddings: (B, seq_length, hidden_size)
        token_mask: (B, seq_length)
        return: per token representation calculated by CNN (B, seq_length, num_filters).
        """
        out = self.doc_cnn(token_embeddings, token_mask)
        return out

    def encode_question(self, question_embeddings, question_token_mask):
        """
        Encodes the question string. Since unlike doc encoder, we need one vector
        for a question, we need to combine the token representations. We currently
        support avging and self attention (Sec 3.2, Question encoding)
        question_embeddings: (B, seq_length, hidden_size)
        question_token_mask: (B, seq_length)
        """
        question_hiddens = self.question_cnn(
            question_embeddings, question_token_mask
        )

        if self.args.question_merge == 'max_pool':
            return self.self_attn(question_hiddens, question_token_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(
                question_hiddens, question_token_mask
            )
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(
                question_hiddens, question_token_mask
            )
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        return question_hiddens, question_hidden


class RnnDocReader(DocReader):
    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__(args, normalize=True)
        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=self.doc_input_size,
            hidden_size=self.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def encode_doc(
        self, token_embeddings, token_mask, question_embeddings, question_mask, question_emb
    ):
        """
        Encode document with multilayer RNN.
        token_embeddings: (B, seq_length, hidden_size)
        token_mask: (B, seq_length)
        return: per token representation calculated by rnn (B, seq_length, hidden_size)
        """
        return self.doc_rnn(token_embeddings, token_mask)

    def encode_question(self, question_embeddings, question_token_mask):
        """
        Encodes the question string. Since unlike doc encoder, we need one vector
        for a question, we need to combine the token representations. We currently
        support avging and self attention (Sec 3.2, Question encoding)
        question_embeddings: (B, seq_length, hidden_size)
        question_token_mask: (B, seq_length)
        """
        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(
            question_embeddings, question_token_mask
        )
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(
                question_hiddens, question_token_mask
            )
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(
                question_hiddens, question_token_mask
            )
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        return question_hiddens, question_hidden


class DataParallelDocReader(nn.DataParallel):
    def __init__(self, doc_reader):
        super().__init__(doc_reader)
        self.doc_reader = doc_reader

    @property
    def embedding(self):
        return self.doc_reader.embedding

    @embedding.setter
    def embedding(self, value):
        self.doc_reader.embedding = value

    @property
    def fixed_embedding(self):
        return self.doc_reader.fixed_embedding

    @fixed_embedding.setter
    def fixed_embedding(self, value):
        self.doc_reader.fixed_embedding = value


class DistributedDataParallelDocReader(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, doc_reader):
        super().__init__(doc_reader)
        self.doc_reader = doc_reader

    @property
    def embedding(self):
        return self.doc_reader.embedding

    @embedding.setter
    def embedding(self, value):
        self.doc_reader.embedding = value

    @property
    def fixed_embedding(self):
        return self.doc_reader.fixed_embedding

    @fixed_embedding.setter
    def fixed_embedding(self, value):
        self.doc_reader.fixed_embedding = value

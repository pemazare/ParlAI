#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import pdb

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
from .rnn_reader import DocReader
from .rnn_reader import EnsembleDocReader
from .rnn_reader import RnnDocReader
from .rnn_reader import CNNDocReader
from .rnn_reader import CNNDocRNNQuestionReader
from .rnn_reader import JengaDocReader
from .rnn_reader import AttentionalDocReader
from .rnn_reader import ResnetDocReader
from .rnn_reader import DistributedDataParallelDocReader, DataParallelDocReader
from . import config
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

logger = logging.getLogger()


class DocReaderModel(object):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, args, word_dict, feature_dict, state_dict=None, normalize=True, character_dict=None):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.character_dict = character_dict
        self.updates = 0
        self.normalize = normalize
        self.train_loss = AverageMeter()

        # Building network.
        # if args.model == 'rnn':
        #     self.network = RnnDocReader(args, normalize)
        # elif args.model == 'cnn':
        #     self.network = CNNDocReader(args, normalize)
        # elif args.model == 'cnn_rnn':
        #     self.network = CNNDocRNNQuestionReader(args, normalize)
        # elif args.model == 'resnet':
        #     self.network = ResnetDocReader(args, normalize)
        # elif args.model == 'jenga':
        self.network = JengaDocReader(args, normalize, character_dict)
        # elif args.model == 'attn':
        #     self.network = AttentionalDocReader(args)
        # else:
        #     raise RuntimeError('Unsupported model: %s' % args.model)
        if not hasattr(args, 'distributed'):
            args.distributed = False
        if args.cuda:
            self.network.cuda()
        if args.distributed:
            self.network = DistributedDataParallelDocReader(self.network)
        elif args.multi_gpu:
            self.network = DataParallelDocReader(self.network)
        if state_dict:
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, args.learning_rate,
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        elif args.optimizer == 'nag':
            self.optimizer = NAG(parameters, args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        lr=args.learning_rate,
                                        weight_decay=args.weight_decay)
        elif args.optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(parameters,
                                           lr=args.learning_rate,
                                           weight_decay=args.weight_decay)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(parameters,
                                           lr=args.learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % args.optimizer)
        if args.use_annealing_schedule:
            # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=5, min_lr=1e-4,mode='max')
            self.lr_scheduler = MultiStepLR(self.optimizer, [30, 60, 80])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def set_embeddings(self):
        # Read word embeddings.
        if not self.args.embedding_file:
            logger.warn('[ WARNING: No embeddings provided. '
                        'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings ]')
        embeddings = load_embeddings(self.args, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.network.embedding.weight.size()
        assert(new_size[1] == old_size[1])
        if new_size[0] != old_size[0]:
            logger.warn('[ WARNING: Number of embeddings changed (%d->%d) ]' %
                        (old_size[0], new_size[0]))

        # Swap weights
        self.network.embedding.weight.data = embeddings

        # If partially tuning the embeddings, keep the old values
        if self.args.tune_partial > 0:
            fixed_embedding = embeddings[self.args.tune_partial + 2:]
            self.network.fixed_embedding = fixed_embedding


    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))
            print('For debug: ', self.args.vocab_size)
            print('Network: ', type(self.network))

            def expand_embeddings(net):
                old_embedding = net.embedding.weight.data
                net.doc_reader.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                            self.args.embedding_dim,
                                                            padding_idx=0)
                new_embedding = net.embedding.weight.data

                print('Old embedding size: ', old_embedding.size())
                print('New embedding size (before add): ', new_embedding.size())
                new_embedding[:old_embedding.size(0)] = old_embedding
                print('New embedding size: ', new_embedding.size())

            if isinstance(self.network, EnsembleDocReader):
                print("Compacting ensemble embeddings")
                expand_embeddings(self.network.networks[0])
                emb_layer = self.network.networks[0].doc_reader.embedding
                for net in self.network.networks:
                    if not hasattr(net.doc_reader.args, 'learn_unk') or not net.doc_reader.args.learn_unk:
                        net.doc_reader.args.learn_unk = True
                        net.doc_reader.unk_index = 1
                        net.doc_reader.unk_emb = Parameter(torch.zeros(self.args.embedding_dim))
                        net.doc_reader.unk_emb.data = net.doc_reader.embedding.weight.data[self.word_dict[self.word_dict.UNK]].clone()
                    if not hasattr(net.doc_reader.args, 'learn_hash_emb'):
                        net.doc_reader.args.learn_hash_emb = 0

                    net.doc_reader.embedding = emb_layer
                print("Setting explicitly to 0 the UNK vector")
                emb_layer.weight.data[self.word_dict[self.word_dict.UNK]].zero_()
            else:
                expand_embeddings(self.network)
        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        if isinstance(self.network, EnsembleDocReader):
            embedding = self.network.networks[0].doc_reader.embedding.weight.data
        else:
            embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file, encoding='UTF-8') as f:
            for i, line in enumerate(f):
                parsed = line.rstrip().split(' ')
                if len(parsed) != embedding.size(1) + 1:
                    if i == 0:
                        logging.warning('Skipping first line of embedding file %s ' % embedding_file)
                        continue
                    else:
                        assert(False)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def prepare_ex(self, ex, volatile=False):
        ex = ex.copy()
        def make_var(x, cuda_if_enabled):
            if torch.is_tensor(x):
                return Variable(
                    x.cuda(async=True) if (self.args.cuda and cuda_if_enabled) else x,
                    volatile=volatile,
                )
            return x
        for k in {'x1', 'x1_f', 'x1_mask', 'x2', 'x2_mask', 'start', 'end', 'para_ids'}:
            if k in ex:
                ex[k] = make_var(ex[k], True)
        for k in {'x1_char', 'x1_char_lengths', 'x2_char', 'x2_char_lengths'}:
            if k in ex:
                ex[k] = make_var(ex[k], False)
        return ex

    def transform_inputs_for_para_mode(self, ex):
        batch_size, num_paras, padded_para_len = ex['x1'].size()
        padded_q_len = ex['x2'].size(1)
        q = ex['x2'].unsqueeze(1).expand(batch_size, num_paras, padded_q_len)
        q_mask = ex['x2_mask'].unsqueeze(1).expand(batch_size, num_paras, padded_q_len)
        docs = ex['x1'].view(-1, padded_para_len) # [batch*num_paras, para_len]
        doc_feats = ex['x1_f'].view(-1, padded_para_len, ex['x1_f'].size(3))
        doc_masks = ex['x1_mask'].view(-1, padded_para_len)
        q = q.contiguous().view(-1, q.size(2))
        q_mask = q_mask.contiguous().view(-1, q_mask.size(2))
        return dict(
            ex,
            x1=docs,
            x1_f=doc_feats,
            x1_mask=doc_masks,
            x2=q,
            x2_mask=q_mask,
        )

    def get_disjoint_scores(self, ex, para_mode):
        #transform the inputs if para_mode
        if para_mode:
            batch_size, num_paras, padded_para_len = ex['x1'].size()
            ex = self.transform_inputs_for_para_mode(ex)
        # Run forward

        score_s, score_e = self.network(ex)
        # # Chunk input during inference to avoid OOM in GPU
        # weight = ex['x1'].size(0) * ex['x1'].size(1) * ex['x2'].size(1)
        # max_weight = 400000
        # chunk_size = max(max_weight // (ex['x1'].size(1) * ex['x2'].size(1)), 1)
        # ex_keys = ex.keys()
        # score_s_list = []
        # score_e_list = []
        # for x1, x1_f, x1_mask, x2, x2_mask in zip(*map(
        #     lambda x: torch.split(ex[x], chunk_size),
        #     ['x1', 'x1_f', 'x1_mask', 'x2', 'x2_mask'],
        # )):
        #     score_s, score_e = self.network({
        #         'x1': x1,
        #         'x1_f': x1_f,
        #         'x1_mask': x1_mask,
        #         'x2': x2,
        #         'x2_mask': x2_mask,
        #     })
        #     score_s_list.append(score_s)
        #     score_e_list.append(score_e)
        # score_s = torch.cat(score_s_list)
        # score_e = torch.cat(score_e_list)

        def mask_scores(x):
            mask = ex['x1_mask']
            if x.dim() == 3:
                mask = mask.unsqueeze(-1).expand_as(x)
            x.data.masked_fill_(mask.data, -float('inf'))
        mask_scores(score_s)
        mask_scores(score_e)

        if para_mode:
            score_s = score_s.contiguous().view(batch_size, num_paras * padded_para_len)
            score_e = score_e.contiguous().view(batch_size, num_paras * padded_para_len)

        return score_s, score_e


    def update(self, ex, para_mode=False):
        # Train mode
        self.network.train()
        # Transfer to GPU
        if self.args.cuda:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        ex_dict = {
                'x1': inputs[0],
                'x1_f': inputs[1],
                'x1_mask': inputs[2],
                'x2': inputs[3],
                'x2_mask': inputs[4],
        }

        if self.args.span_joint_optimization:
            if para_mode:
                raise NotImplementedError
            if (target_e.data - target_s.data).max() + 1 > self.args.max_len:
                logger.warning("Skipped a line with len=%d > max_len=%d" %
                    ((target_e.data - target_s.data).max() + 1, self.args.max_len))
                return 0, 0
            joint_score = self.network(ex_dict) # [B, (doc_size * max_len)]
            joint_target = target_s + ex_dict['x1'].size(1) * (target_e - target_s)
            # Note that this needs to be in sync with model.py:decode_joint
            # print('target:', joint_target)
            # print('s: ', target_s)
            # print('e: ', target_e)
            # print('score size: ', joint_score.size())
            loss = F.nll_loss(joint_score, joint_target)
        else:
            score_s, score_e = self.get_disjoint_scores(ex_dict, para_mode)
            score_s, score_e = F.log_softmax(score_s, dim=1), F.log_softmax(score_e, dim=1)
            if score_s.dim() == 3:
                assert(score_e.dim() == 3)
                score_s = score_s.sum(dim=-1) / score_s.size(-1)
                score_e = score_e.sum(dim=-1) / score_e.size(-1)


            # Compute loss and accuracies
            def ranking_loss(scores, target):
                target_scores = scores[torch.arange(0, scores.size(0)).cuda(async=True).long(), target.data]
                losses = (self.args.ranking_loss_margin + scores - target_scores.unsqueeze(1)).clamp(min=0)
                # Adjust formula because target position is always a violator and creates a loss equal to the margin
                num_violators = ((losses > 0).long().sum(dim=1) - 1).clamp(min=1) # to avoid division by 0 (if no violators, loss is 0 anyway)
                return ((losses.sum(dim=1) - self.args.ranking_loss_margin) / num_violators.float()).mean()
            loss_fn = ranking_loss if self.args.ranking_loss else F.nll_loss
            loss = loss_fn(score_s, target_s) + loss_fn(score_e, target_e)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()
        self.train_loss.update(loss.data[0], ex[0].size(0))


        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex_dict['x1'].size(0)

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, para_mode=None, candidates=None, top_n=1, async_pool=None, ensemble=None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.args.cuda:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:5]]
        ex_dict = {
                'x1': inputs[0],
                'x1_f': inputs[1],
                'x1_mask': inputs[2],
                'x2': inputs[3],
                'x2_mask': inputs[4],
        }
        text = ex[-2]
        spans = ex[-1]
        if ensemble:
            if para_mode:
                raise NotImplementedError
            score_s_list = []
            score_e_list = []
            for network in self.network.networks:
                if network.permutation_required:
                    ex = ex.copy()
                    ex['x1'] = network.permutation.index_select(0, ex['x1'].view(-1)).view(ex['x1'].size())
                    ex['x2'] = network.permutation.index_select(0, ex['x2'].view(-1)).view(ex['x2'].size())
                score_s, score_e = network(ex)
                score_s_list.append(score_s)
                score_e_list.append(score_e)
            score_s = torch.stack(score_s_list, dim=1)  # [batch, num_models, c_len]
            score_e = torch.stack(score_e_list, dim=1)  # [batch, num_models, c_len]
            score_s, score_e = F.softmax(score_s, dim=2), F.softmax(score_e, dim=2)
        else:
            # Run forward
            # TODO support ensemble mode
            if self.args.span_joint_optimization:
                if para_mode:
                    raise NotImplementedError
                joint_score = self.network(ex)
                joint_score = joint_score.data.cpu()
                args = (joint_score, top_n, self.args.max_len)
                if async_pool:
                    return async_pool.apply_async(self.decode_joint, args)
                else:
                    return self.decode_joint(*args)
            score_s, score_e = self.get_disjoint_scores(ex_dict, para_mode)
            if self.normalize:
                score_s, score_e = F.softmax(score_s, dim=1), F.softmax(score_e, dim=1)
                if score_s.dim() == 3:
                    assert(score_e.dim() == 3)
                    score_s = score_s.prod(dim=-1)
                    score_e = score_e.prod(dim=-1)
            else:
                # score_s, score_e = F.sigmoid(score_s), F.sigmoid(score_e)
                score_s, score_e = score_s.exp(), score_e.exp()
        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        para_len = ex_dict['x1'].size(2) if para_mode else None

        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len, para_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len, para_len, text, spans)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)


    @staticmethod
    def decode_joint(joint_score, top_n=1, max_len=None):
        pred_s = []
        pred_e = []
        pred_score = []

        # [batch, (max_len * c_size)]
        for i in range(joint_score.size(0)):
            # compute a matrix of scores of size (max_len, score_s.size(1))
            # where the index along the first dimension represents the answer
            # length (starting from 1) and the index along the second dimension
            # is the offset in the document
            scores = joint_score[i].view(max_len, -1)
            c_size = scores.size(1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx = [x % c_size for x in idx_sort]
            e_idx = [s_idx[i] + (idx_sort[i] // c_size) for i in range(len(idx_sort))]
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score


    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None, para_len=None, text=None, spans=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        para_id = []
        c_len = score_s.size(-1)
        max_len = max_len or c_len

        # For each example, compute a matrix of scores of size (max_len, c_len)
        # where the index along the first dimension represents the answer
        # length (starting from 1) and the index along the second dimension
        # is the offset in the document

        # Repeat score_s max_len times in a new second-to-last dimension
        # (we need to combine each starting score with its max_len possible corresponding end scores)
        base_size = list(score_s.size())[:-1]
        scores = score_s.unsqueeze(-2)  # [batch(, num_models), 1, c_len]
        scores = scores.expand(*(base_size + [max_len, c_len])).contiguous()  # [batch(, num_models), max_len, c_len]

        # Build tensor of end indices
        score_e_indices = torch.arange(0, c_len).long().unsqueeze(0).expand(max_len, c_len) + \
            torch.arange(0, max_len).long().unsqueeze(1).expand(max_len, c_len)
        # replace out-of-bounds indices with special index c_len
        mask = score_e_indices >= c_len
        score_e_indices.masked_fill_(mask, c_len)

        # Pad score_e with zeros so that the index c_len maps to 0
        padded_score_e = torch.cat([score_e, score_e.new(*(base_size + [1])).zero_()], dim=-1)

        # Multiply each start score with the corresponding end score
        scores *= padded_score_e \
            .index_select(-1, score_e_indices.view(-1)) \
            .view(scores.size()) \
        # scores is now of size [batch(, num_models), max_len, c_len]

        if scores.dim() == 4:
            # ensemble model: sum along model direction
            scores = scores.sum(dim=1)  # [batch, max_len, c_len]
        scores = scores.numpy()
        predictions = []
        for i in range(score_s.size(0)):
            # Take argmax or top n
            scores_flat = scores[i].flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx = [x % c_len for x in idx_sort]
            e_idx = [s_idx[i] + idx_sort[i] // c_len for i in range(len(idx_sort))]
            s_offset, e_offset = spans[i][s_idx[0]][0], spans[i][e_idx[0]][1]
            predictions.append(text[i][s_offset:e_offset])

            pred_score.append(scores_flat[idx_sort])
            if para_len is None:
                pred_s.append(s_idx)
                pred_e.append(e_idx)
            else:
                pred_s.append([x % para_len for x in s_idx])
                pred_e.append([x % para_len for x in e_idx])
                para_id.append([x // para_len for x in s_idx])
        return predictions

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None, para_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        pred_para_id = []
        para_mode = para_len is not None
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx, para_idx = [], [], [], []

            def append_matches(tokens, para_id=None):
                for s, e in tokens.ngrams(n=max_len, as_strings=False):
                    span = tokens.slice(s, e).untokenize()
                    if span in cands or span.lower() in cands:
                        # Match! Record its score.
                        scores.append(score_s[i][s] * score_e[i][e - 1])
                        s_idx.append(s)
                        e_idx.append(e - 1)
                        if para_id is not None:
                            para_idx.append(para_id)

            if para_mode:
                for para_id, para_tokens in enumerate(tokens):
                    append_matches(para_tokens, para_id)
            else:
                append_matches(tokens)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
                if para_mode:
                    para_idx.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)
                para_idx = np.array(para_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
                if para_mode:
                    pred_para_id.append(para_idx[idx_sort])
        if para_mode:
            return pred_s, pred_e, pred_score, pred_para_id
        else:
            return pred_s, pred_e, pred_score

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            offset = self.args.tune_partial + 2
            self.network.embedding.weight.data[offset:] \
                = self.network.fixed_embedding

    def save(self, filename, epoch=None):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'config': vars(self.args),
        }
        if hasattr(self, 'character_dict'):
            params['character_dict'] = self.character_dict
        if epoch:
            params['epoch'] = epoch
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(filename, map_location=lambda storage, loc: storage)
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        saved_config = saved_params['config']
        default = config.parse_args([])
        for key in vars(default):
            if key not in saved_config:
                saved_config[key] = getattr(default, key)
        character_dict = saved_params.get('character_dict')
        args = argparse.Namespace(**saved_config)
        reader_model = DocReaderModel(args, word_dict, feature_dict, state_dict=state_dict, normalize=normalize, character_dict=character_dict)
        return reader_model

    def cuda(self):
        self.network.cuda()

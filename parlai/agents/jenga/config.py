# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import logging
import os


logger = logging.getLogger()

USER = os.getenv('USER') or 'root'
DATA_DIR = '/private/home/raison/data'
EMBED_DIR = '/private/home/raison/embeddings'
MODEL_DIR = '/checkpoint/' + USER + '/squad/rnn/adhoc'


def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('DrQA Arguments')
    agent.add_argument('--no_cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--data_workers', type=int, default=5)
    agent.add_argument('--multi_gpu', type='bool', default=False)
    agent.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    agent.add_argument('--dist_url', type=str, default='env://')
    agent.add_argument('--dist_backend', type=str, default='gloo')
    # Basics
    agent.add_argument('--model_dir', type=str, default=MODEL_DIR)
    agent.add_argument('--model_name', type=str, default=None)
    agent.add_argument('--checkpoint', type='bool', default=True)
    agent.add_argument('--random_seed', type=int, default=1013)
    agent.add_argument('--uncased_question', type='bool', default=False)
    agent.add_argument('--uncased_doc', type='bool', default=False)

    # Data files
    agent.add_argument('--data_dir', type=str, default=DATA_DIR)
    agent.add_argument('--embed_dir', type=str, default=EMBED_DIR)
    agent.add_argument('--eval_file', type=str, default='doc-dev-v1.1.json')
    agent.add_argument(
        '--train_file', type=str, default='doc-train-processed-spacy.txt'
    )
    agent.add_argument(
        '--dev_file', type=str, default='doc-dev-processed-spacy.txt'
    )
    agent.add_argument(
        '--embedding_file', type=str, default=None
    )
    agent.add_argument(
        '--pretrained', type=str, default=None, help='Pre-trained model'
    )
    agent.add_argument(
        '--unlabeled',
        type='bool',
        default=False,
        help='Data is unlabeled (prediction only)'
    )
    agent.add_argument('--skip_no_answer', type='bool', default=True)
    agent.add_argument('--scale_gradients', type='bool', default=False)
    # Model details
    # agent.add_argument(
    #     '--model', type=str, default='jenga', help='Model architecture type'
    # )
    agent.add_argument('--vocab_size', type=int, default=None)
    agent.add_argument('--num_features', type=int, default=None)
    agent.add_argument(
        '--fix_embeddings',
        type='bool',
        default=True,
        help='Keep word embeddings fixed (pretrained)'
    )
    agent.add_argument('--learn_unk', type='bool', default=False)
    agent.add_argument('--learn_hash_emb', type=int, default=0)
    agent.add_argument('--tune_partial', type=int, default=0)
    agent.add_argument(
        '--embedding_dim',
        type=int,
        default=None,
        help=('Default embedding size if '
              'embedding_file is not given')
    )
    agent.add_argument(
        '--hidden_size', type=int, default=64, help='Hidden size of RNN units'
    )
    agent.add_argument(
        '--doc_layers',
        type=int,
        default=3,
        help='Number of RNN layers for passage'
    )
    agent.add_argument(
        '--question_layers',
        type=int,
        default=3,
        help='Number of RNN layers for question'
    )
    agent.add_argument(
        '--rnn_type',
        type=str,
        default='lstm',
        help='RNN type: lstm (default), gru, rnn or sru'
    )
    # cnn specific args
    agent.add_argument(
        '--num_filters',
        type=str,
        default='64,64',
        help='Number of filters'
    )
    agent.add_argument(
        '--kernel_widths',
        type=str,
        default='1,5',
        help='kernel_width of the filters'
    )
    agent.add_argument(
        '--cnn_doc_layers',
        type=int,
        default=12,
        help='Number of CNN layers for doc'
    )
    agent.add_argument(
        '--cnn_question_layers',
        type=int,
        default=3,
        help='Number of CNN layers for question'
    )
    agent.add_argument('--concat_cnn_layers', type='bool', default=False)
    agent.add_argument(
        '--dropout_cnn',
        type=float,
        default=0.3,
        help='Dropout rate for CNN states'
    )
    agent.add_argument(
        '--fast_debug',
        type='bool',
        default=False,
        help='Loads only top 1000 embeddings to speed up start time'
    )
    agent.add_argument(
        '--dropout_jenga',
        type=float,
        default=0.3,
        help='Dropout rate for Jenga layers'
    )
    agent.add_argument(
        '--jenga_word_dropout',
        type=float,
        default=0,
        help='Word dropout rate for Jenga layers'
    )
    agent.add_argument(
        '--jenga_fix_word_dropout',
        type='bool',
        default=False,
        help='Fix word dropout behavior for Jenga layers'
    )
    agent.add_argument(
        '--jenga_self_attention',
        type='bool',
        default=False,
        help='Add self attention to Jenga RNNs'
    )
    agent.add_argument(
        '--jenga_attention_cat',
        type='bool',
        default=False,
        help='Remove diagonal from attention score matrix and concat attention w/ input'
    )
    agent.add_argument(
        '--jenga_attention_residuals',
        type='bool',
        default=False,
        help='Residuals in jenga attention'
    )
    agent.add_argument(
        '--jenga_residual_rnns',
        type='bool',
        default=False,
        help='Residuals in jenga'
    )
    agent.add_argument(
        '--jenga_nb_attention_head',
        type=int,
        default=1,
        help='Number of self attention heads in Jenga'
    )
    agent.add_argument(
        '--jenga_scale_attention_output',
        type='bool',
        default=False,
        help='Whether to scale attention dot-product by 1/sqrt(ndim)'
    )
    agent.add_argument(
        '--jenga_attention_extra_scaling',
        type=int,
        default=None,
        help='Extra factor to scale attention'
    )
    agent.add_argument(
        '--jenga_attention_projected',
        type='bool',
        default=False,
        help='Whether to return the projected value of the attention. Must be true for multiple attention heads'
    )
    agent.add_argument(
        '--jenga_lf_dropout',
        type=float,
        default=0,
        help='Dropout rate for language features'
    )
    agent.add_argument(
        '--jenga_emb_diff_mode',
        type=int,
        default=0,
        help='0: do nothing. 1: replace context with context - question. 2: replace context with abs(context - question). Negative: same thing but switch question and context'
    )
    agent.add_argument(
        '--jenga_q_c_dot_product',
        type='bool',
        default=False,
        help='Add a continuous feature with the dot-product between quesiton and context words'
    )
    agent.add_argument(
        '--jenga_flatten_question',
        type='bool',
        default=False,
        help='Run a max-pooling question-wise before applying the RNN on the context'
    )
    agent.add_argument(
        '--jenga_use_question_final',
        type='bool',
        default=False,
        help='Use the question at the last stage of jenga'
    )
    agent.add_argument(
        '--jenga_iterative_decode',
        type='bool',
        default=False,
        help='Iterate over output vector. Requires jenga_use_question_final=True to work'
    )
    agent.add_argument(
        '--dropout_2d_filter',
        type=bool,
        default=True,
        help='Use 2D dropout to drop filters randomly'
    )
    agent.add_argument(
        '--dropout_words',
        type='bool',
        default=True,
        help='Dropout words from sequence randomly at the input layer.'
    )
    agent.add_argument(
        '--use_dropout_2d_filter',
        type='bool',
        default=True,
        help='Dropout filters.'
    )
    agent.add_argument(
        '--use_word_dropout_all_layers',
        type='bool',
        default=False,
        help='Dropout words from sequence randomly at all layers'
    )
    agent.add_argument(
        '--dropout_cnn_output',
        type='bool',
        default=True,
        help='Whether to dropout the CNN output'
    )
    agent.add_argument(
        '--use_dilation',
        type='bool',
        default=False,
        help='Whether to use dilation'
    )
    agent.add_argument(
        '--use_gated_cnns',
        type='bool',
        default=True,
        help='Whether to use gated cnns'
    )
    agent.add_argument(
        '--use_residual_layer',
        type='bool',
        default=True,
        help='Add residual connection'
    )
    agent.add_argument(
        '--use_gated_question_attention',
        type='bool',
        default=True,
        help='Add gates to the question attention'
    )
    agent.add_argument(
        '--qemb_project_before_softmax',
        type='bool',
        default=False,
        help='Question matching projects RNN hidden state to different space'
    )
    agent.add_argument(
        '--qemb_match_first_layer',
        type='bool',
        default=False,
        help='Question matching on the input layer'
    )
    agent.add_argument(
        '--qemb_match_all_layers',
        type='bool',
        default=True,
        help='Add gates to the question attention'
    )
    agent.add_argument(
        '--qemb_match_last_layer',
        type='bool',
        default=False,
        help='Add gates to the question attention'
    )
    agent.add_argument(
        '--use_gated_question_attention_input',
        type='bool',
        default=True,
        help='Add gates to the question attention'
    )
    agent.add_argument(
        '--paragraph_self_attn',
        type='bool',
        default=False,
        help='Computing attention between a paragraph word with all other words'
    )
    agent.add_argument(
        '--attn_type',
        type=str,
        default='dot',
        help='type of attention, mlp or dot product'
    )
    agent.add_argument(
        '--share_gates_param',
        type='bool',
        default=True,
        help='Add gates to the question attention'
    )

    agent.add_argument(
        '--pool_context_output_layer',
        type='bool',
        default=False,
        help='Before predicting the start or end span, pool nearby context'
    )
    agent.add_argument(
        '--pool_type',
        type=str,
        default='concat',
        help='Type of pooling, avg or max pool or concat'
    )
    agent.add_argument(
        '--context_size', type=int, default=5, help='Context to pool'
    )
    agent.add_argument(
        '--dropout_pooled_context',
        type=float,
        default=0.2,
        help='dropout after pooling the context'
    )
    agent.add_argument(
        '--para_mode',
        type='bool',
        default=False,
        help='represent a doc as a list of paras instead of a huge list of words'
    )
    agent.add_argument(
        '--max_num_paras',
        type=int,
        default=10,
        help='1 + number of negative paragraphs to consider during training. This number also includes the positive para (hence the 1+)'
    )

    # Optimization details
    agent.add_argument(
        '--max_len',
        type=int,
        default=15,
        help='The max span allowed during decoding'
    )
    agent.add_argument(
        '--valid_metric',
        type=str,
        default='f1',
        help='The evaluation metric used for model selection'
    )
    agent.add_argument(
        '--batch_size', type=int, default=10, help='Batch size (default 32)'
    )
    agent.add_argument(
        '--test_batch_size',
        type=int,
        default=10,
        help='Test batch size (default 128)'
    )
    agent.add_argument('--rnn_padding', type='bool', default=False)
    agent.add_argument('--start_epoch', type=int, default=0)
    agent.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs (default 40)'
    )
    agent.add_argument(
        '--display_iter',
        type=int,
        default=25,
        help='Print train error after every \
                              <display_iter> epoches (default 10)'
    )
    agent.add_argument(
        '--dropout_emb',
        type=float,
        default=0.1,
        help='Dropout rate for word embeddings'
    )
    agent.add_argument(
        '--dropout_rnn',
        type=float,
        default=0.4,
        help='Dropout rate for RNN states'
    )
    agent.add_argument(
        '--dropout_rnn_output',
        type='bool',
        default=True,
        help='Whether to dropout the RNN output'
    )
    agent.add_argument(
        '--optimizer',
        type=str,
        default='adamax',
        help='Optimizer: sgd or adamax (default)'
    )
    agent.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=0.002,
        help='Learning rate for SGD (default 0.002)'
    )
    agent.add_argument(
        '--grad_clipping',
        type=float,
        default=10,
        help='Gradient clipping (default 10.0)'
    )
    agent.add_argument(
	'--use_annealing_schedule',
	type='bool',
	default=True,
	help='Whether to use an annealing schedule or not.'
    )
    agent.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay (default 0)'
    )
    agent.add_argument(
        '--momentum', type=float, default=0, help='Momentum (default 0)'
    )

    # Model-specific
    agent.add_argument('--concat_rnn_layers', type='bool', default=False)
    agent.add_argument(
        '--question_merge',
        type=str,
        default='self_attn',
        help='The way of computing question representation'
    )
    agent.add_argument(
        '--use_qemb',
        type='bool',
        default=True,
        help='Whether to use weighted question embeddings'
    )
    agent.add_argument(
        '--use_in_question',
        type='bool',
        default=True,
        help='Whether to use in_question features'
    )
    agent.add_argument(
        '--use_pos',
        type='bool',
        default=False,
        help='Whether to use pos features'
    )
    agent.add_argument(
        '--use_ner',
        type='bool',
        default=False,
        help='Whether to use ner features'
    )
    agent.add_argument(
        '--use_lemma',
        type='bool',
        default=False,
        help='Whether to use lemma features'
    )
    agent.add_argument(
        '--use_tf',
        type='bool',
        default=True,
        help='Whether to use tf features'
    )
    agent.add_argument(
        '--jenga_use_start_for_end',
        type='bool',
        default=False,
        help='Use start features for end prediction'
    )
    agent.add_argument(
        '--jenga_language_features',
        type='bool',
        default=False,
        help='Whether to allow language features in jenga'
    )
    agent.add_argument(
        '--span_joint_optimization',
        type='bool',
        default=False,
        help='Enable joint optimization of start and end of span'
    )
    agent.add_argument(
        '--length_in_jo',
        type='bool',
        default=False,
        help='Enable length feature in joint optimization of start and end of span'
    )
    agent.add_argument(
        '--jo_projection_hidden_size',
        type=int,
        default=None,
        help='Hidden size on which to project for joint optim. If none, we do not project and directly predict a probability'
    )
    agent.add_argument(
        '--jo_bilinear',
        type='bool',
        default=False,
        help='Use a bilinear projection for joint start_end optimization'
    )
    agent.add_argument(
        '--sort_by_len',
        type='bool',
        default=True,
        help='Whether to sort the examples by document length'
    )
    agent.add_argument(
        '--blocks',
        type=int,
        default=9,
        help='Number of jenga/resnet blocks to use'
    )
    agent.add_argument(
        '--jenga_train_last_layer_only',
        type='bool',
        default=False,
        help='Train only the last layer of an existing model (use with --pretrained)'
    )
    agent.add_argument(
        '--jenga_shortcut_embedding',
        type='bool',
        default=False,
        help='Adds a shortcut connection from the doc embedding to the final max-pooling'
    )
    agent.add_argument(
        '--jenga_arch',
        type=str,
        default='q_first',
        help='Ordering of jenga layers ("q_first", "c_first", "concat" or "grid")'
    )
    agent.add_argument(
        '--jenga_cnn_start_11',
        type='bool',
        default=False,
        help='Whether to start with a 1-wide convolution in jenga CNN'
    )
    agent.add_argument(
        '--jenga_cnn',
        type=str,
        default='off',
        help='Whether to replace RNNs by CNNs in Jenga. "off", "1d", "2d"'
    )
    agent.add_argument(
        '--jenga_cnn_gating',
        type='bool',
        default=False,
        help='Enable gating for Jenga CNN'
    )
    agent.add_argument(
        '--jenga_kernel_width',
        type=int,
        default=3,
        help='Jenga kernel widths'
    )
    agent.add_argument(
        '--jenga_do_b4_res',
        type='bool',
        default=False,
        help='Apply jenga dropout before res connections'
    )
    agent.add_argument(
        '--jenga_remove_last_do',
        type='bool',
        default=False,
        help='Remove last dropout before maxpool'
    )
    agent.add_argument(
        '--jenga_fix_residuals',
        type='bool',
        default=False,
        help='Re-normalize residual connections'
    )
    agent.add_argument(
        '--max_context_len',
        type=int,
        help='Ignore training examples with longer context length'
    )
    agent.add_argument(
        '--max_para_len',
        type=int,
        help='Ignore training examples with longer paragraph length'
    )
    agent.add_argument(
        '--max_question_len',
        type=int,
        help='Ignore training examples with longer question length'
    )
    agent.add_argument(
        '--max_validation_examples',
        type=int,
        help='Cap number of examples to validate on'
    )
    agent.add_argument(
        '--character_level',
        type='bool',
        default=False,
        help='Also represents tokens with a character level RNN (Jenga only)'
    )
    agent.add_argument(
        '--character_level_size',
        type=int,
        default=150,
        help='Output size of character-level RNN'
    )
    agent.add_argument(
        '--ranking_loss',
        type='bool',
        default=False,
        help='Whether to use a ranking loss instead of NLL',
    )
    agent.add_argument(
        '--ranking_loss_margin',
        type=float,
        default=0.1,
        help='Margin for ranking loss',
    )
    agent.add_argument(
        '--add_context_maxpool',
        type='bool',
        default=False,
        help='Add maxpool over all context at each layer',
    )
    agent.add_argument(
        '--grid_horizontal_only',
        type='bool',
        default=False,
    )
    agent.add_argument(
        '--grid_single_direction',
        type='bool',
        default=False,
    )
    agent.add_argument(
        '--custom_lstm_bias_init',
        type='bool',
        default=False,
    )

    # Print predictions
    agent.add_argument('--only_print_predictions', type='bool', default=False)
    agent.add_argument('--only_validate', type='bool', default=False)
    agent.add_argument(
        '--ensemble', type=str, default=None, help='Pre-trained model'
    )
    agent.add_argument(
        '--top_n', type=int, default=1, help='Pick best prediction among top n'
    )

def set_defaults(opt):
    # Embeddings options
    if opt.get('embedding_file'):
        if not os.path.isfile(opt['embedding_file']):
            raise IOError('No such file: %s' % opt['embedding_file'])
        with open(opt['embedding_file']) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        opt['embedding_dim'] = dim
    elif not opt.get('embedding_dim'):
        raise RuntimeError(('Either embedding_file or embedding_dim '
                            'needs to be specified.'))

    # Make sure tune_partial and fix_embeddings are consistent
    if opt['tune_partial'] > 0 and opt['fix_embeddings']:
        print('Setting fix_embeddings to False as tune_partial > 0.')
        opt['fix_embeddings'] = False

    # Make sure fix_embeddings and embedding_file are consistent
    if opt['fix_embeddings']:
        if not opt.get('embedding_file') and not opt.get('pretrained_model'):
            print('Setting fix_embeddings to False as embeddings are random.')
            opt['fix_embeddings'] = False

def override_args(opt, override_opt):
    # Major model args are reset to the values in override_opt.
    # Non-architecture args (like dropout) are kept.
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'doc_layers',
                'question_layers', 'rnn_type', 'optimizer', 'concat_rnn_layers',
                'question_merge', 'use_qemb', 'use_in_question', 'use_tf',
                'vocab_size', 'num_features', 'use_time'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v

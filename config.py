
config = {
    'max_seq_len': 60,
    'max_word_num': 11000,
    'common_word': 0,
    'reserve_common_word': True,
    'max_word_len': 15,
    'reserve_rare_word': True,
    'word_padding': 1,
    'char_padding': 1,
    'val_ratio': 0.1,
    'split_seed': 9788,
    'data_dir': None,
    'log_dir': None,

    'batch_size': 32,

    'use_word': True,
        'word_embedding_dim':30,

    'use_char': False,
        'char_vocabulary_size': 'to be calculated',
        'char_embedding_dim':16,
        'char_feature_maps': [50, 100, 150, 200, 200, 200, 200],
        'char_kernels': [1, 2, 3, 4, 5, 6, 7],
        'char_stddev': 0.02,

    'use_highway': True,
        'highway_layers': 3,

    'use_LSTM': True,
        'LSTM_dim': 50,
        'LSTM_layer_size': 3,
        'fw_forget_bias': 1.0,
        'bw_forget_bias': 1.0,

    'use_CNN': False,
        'seq_feature_maps': [50, 50, 50, 100, 100, 100, 100],
        'seq_kernels': [1, 2, 4, 8, 16, 24, 32],
        'seq_stddev': 0.02,

    'use_direct_embed': False,
        'resize_to': 1000,

    'hidden_dims': [200, 100, 10, 1],

    'learning_rate': 1e-3,
    'n_epoch': 200,
    'print_iteration': 10,

    'epsilon': 1e-5,
}
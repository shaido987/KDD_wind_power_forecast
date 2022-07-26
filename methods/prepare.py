def prep_env():
    settings = {
        "path_to_test_x": "./data/new_tests/test_x",
        "path_to_test_y": "./data/new_tests/test_y",
        "data_path": "./data",
        "filename": "wtbdata_245days.csv",
        "location_file": "sdwpf_baidukddcup2022_turb_location.csv",
        "start_col": 3,
        "checkpoints": "checkpoints",
        "pred_file": "predict.py",
        "framework": "pytorch",
        "device": "cuda:0",
        "is_debug": True,
        'mdlinear': {
            "model_id": "50_alr3_time_split_72step_FE6_abs",
            "seed": 2021,

            # model settings
            "seq_len": 50,
            "pred_len": 288,
            "label_len": 0,
            "target": "Patv",
            "horizons": [72, 144, 216, 288],
            "step_size": 72,
            "enc_in": 7,  # number of features
            "individual": False,
            "moving_avg": 25,  # window size of the moving average
            "scale": True,

            # experiment settings
            "train_epochs": 100,
            "batch_size": 32,
            "weight_decay": 0.0001,
            "learning_rate": 0.001,
            "lradj": "3",  # adjustable learning rate
            "patience": 3,
            "num_workers": 16,
        },
        'xtgn': {
            "model_name": "xtgn-2",
            "seed": 2022,

            # data
            "feature_dim": 4,
            "num_nodes": 134,
            "seq_length_x": 288,
            "seq_length_y": 288,
            "embed_dim": 2,

            # experiment settings
            "batch_size": 32,
            "max_epoch": 100,
            "milestone": [5, 20, 40, 70],
            "learning_rate": 0.003,
            "dropout_rate": 0.2,
            "weight_decay": 0.0001,
            "print_freq": 5,
            "test_time": 5,
            "ratio": 9,  # time split ratio

            # WaveNet model
            "blocks": 2,
            "wavenet_layers": 7,
            "kernel_size": 2,
            "residual_channels": 32,
            "dilation_channels": 32,
            "skip_channels": 128,
            "end_channels": 256,
            "receptive_field": 13,
        }
    }

    return settings

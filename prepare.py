def prep_env():
    settings = {
        "path_to_test_x": "",
        "path_to_test_y": "",
        "data_path": "./../../data",
        "filename": "wtbdata_245days.csv",
        "location_file": "sdwpf_baidukddcup2022_turb_location.csv",
        "start_col": 3,
        "checkpoints": "",
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": True,
        'dlin': {
            "input_len": 50,
            "output_len": 288,
            "label_len": 0,
            "use_gpu": True,
            "gpu": 0,
            "model_id": "50_alr3_time_split_72step_FE6_abs",
            "target": "Patv",
            "checkpoints": "dlin/checkpoints",
            "horizons": [72, 144, 216, 288],
            "step_size": 72,
            "enc_in": 7,  # number of features
            "individual": False,
            "weight_decay": 0.0001,  # L2 regularization
            "moving_avg": 25,  # window size of the moving average
            "scale": True,
            "patience": 3,
            "learning_rate": 0.001,
            "lradj": "3",  # adjustable learning rate
            "train_epochs": 100,
            "batch_size": 32,
            "num_workers": 16,
        },
        'gtcn': {
            "npz_path": "data",
            "checkpoints": "gtcn/model_save",
            "model_name": "test",  # TODO
            "pred_file": "predict.py",
            "framework": "pytorch",
            "is_debug": True,
            "start_col": 3,
            "seed": 2022,
            "device": "cuda:0",

            # data
            "feature_dim": 4,
            "num_nodes": 134,
            "lag": 288,
            "horizon": 288,
            "embed_dim": 2,

            # experiment settings
            "batch_size": 32,
            "max_epoch": 100,
            "milestone": [5, 20, 40, 70],
            "gamma": 0.3,
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "weight_decay": 0.0001,
            "clip": 5,
            "print_freq": 5,
            "test_time": 5,

            # WaveNet model
            "blocks": 2,
            "wavenet_layer": 7,
            "kernel_size": 2,
            "residual_channels": 32,
            "dilation_channels": 32,
            "skip_channels": 128,
            "end_channels": 256,
            "receptive_field": 13,
        }
    }

    # Some duplicate names that are used in the code.
    settings['dlin']["seq_len"] = settings['dlin']["input_len"]
    settings['dlin']["pred_len"] = settings['dlin']["output_len"]
    settings['gtcn']['seq_length_x'] = settings['gtcn']['lag']
    settings['gtcn']['seq_length_y'] = settings['gtcn']['horizon']
    settings['gtcn']['layers'] = settings['gtcn']['wavenet_layer']
    return settings

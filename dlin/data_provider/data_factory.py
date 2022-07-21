import os
from torch.utils.data import DataLoader

from dlin.data_provider.data_loader import Dataset_Custom, Dataset_Pred


def data_provider(args, flag):
    Data = Dataset_Custom

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args['batch_size']
        file = os.path.join(args['data_path'], args['filename'])
        scaler = None
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
        file = args['path_to_test_x']
        scaler = args['scaler']
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']
        file = os.path.join(args['data_path'], args['filename'])
        scaler = None

    data_set = Data(
        data_path=file,
        flag=flag,
        size=[args['seq_len'], args['label_len'], args['pred_len']],
        target=args['target'],
        scale=args['scale'],
        scaler=scaler,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args['num_workers'],
        drop_last=drop_last)
    return data_set, data_loader

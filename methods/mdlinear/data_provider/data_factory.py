import os
from torch.utils.data import DataLoader

from methods.mdlinear.data_provider.data_loader import Dataset_Custom, Dataset_Pred


def data_provider(args, flag):
    loc_file = os.path.join(args['data_path'], args['location_file'])

    if flag == 'pred':
        Data = Dataset_Pred
        data_file = args['path_to_test_x']
        batch_size = 1
        scaler = args['scaler']
        drop_last = False
        shuffle_flag = False
    else:
        # train, val or test
        Data = Dataset_Custom
        data_file = os.path.join(args['data_path'], args['filename'])
        batch_size = args['batch_size']
        scaler = None
        drop_last = True
        shuffle_flag = False if flag == 'test' else True

    data_set = Data(
        data_file=data_file,
        loc_file=loc_file,
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

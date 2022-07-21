import os
import numpy as np
import pandas as pd
from copy import deepcopy


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class TestData(object):
    """
        Desc: Test Data
    """

    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,  # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data = deepcopy(self.df_raw)
        # self.df_data.replace(to_replace=np.nan, value=0, inplace=True)
        self.df_data = self.parse_data(self.df_data)

    def parse_data(self, df):
        # Remove incorrect or unknown values
        nan_rows = df.isnull().any(axis=1)
        invalid_cond = (df['Patv'] < 0) | \
                       ((df['Patv'] == 0) & (df['Wspd'] > 2.5)) | \
                       ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)) | \
                       ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
                        (df['Ndir'] > 720))
        df.loc[invalid_cond, set(df.columns) - {'TurbID', 'Day', 'Tmstamp'}] = np.nan
        df['mask'] = np.where(invalid_cond | nan_rows, 0, 1)


        df.loc[(df['Prtv'] < 0) & (df['Patv'] > 0), 'Prtv'] = np.nan
        df.loc[(df['Prtv'] < 0) & (df['Patv'] <= 0), 'Prtv'] = 0
        df.loc[df['Patv'] < 0, 'Patv'] = 0

        df = df.groupby(df.TurbID.values).ffill().bfill()
        df = self.construct_features(df)
        return df

    def construct_features(self, df):
        # Merge Pab features
        df['Pab_max'] = df[['Pab1', 'Pab2', 'Pab3']].max(axis=1)
        df = df.drop(columns=['Pab1', 'Pab2', 'Pab3'])

        # Drop some features
        df = df.drop(columns=['Etmp', 'Itmp', 'Wdir', 'Ndir'])

        # Make sure the target column is last
        df = df[[c for c in df if c not in ['Patv']] + ['Patv']]

        # Replace any remaining nans with 0.
        df.replace(to_replace=np.nan, value=0, inplace=True)
        return df

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size

        cols = self.df_data.columns[self.start_col:].drop('mask')
        data = self.df_data[cols]

        seq = data.values[border1:border2]
        df = self.df_raw[border1:border2]
        mask = self.df_data['mask'].values[border1:border2]
        return seq, df, mask

    def get_all_turbines(self):
        seqs, dfs, masks = [], [], []
        for i in range(self.farm_capacity):
            seq, df, mask = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
            masks.append(mask)
        return seqs, dfs, masks


def Add_Window_Horizon(data, window=288, step_size=10, horizon=288, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      # windows
    Y = []      # horizon
    M = []
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window, :, :-1])
            Y.append(data[index+window+horizon-1:index+window+horizon, :, :-1])
            M.append(data[index+window+horizon-1:index+window+horizon, :, -1:])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window, :, :-1])
            Y.append(data[index+window:index+window+horizon, :, :-1])
            M.append(data[index+window:index+window+horizon, :, -1:])
            index = index + step_size

    X = np.array(X)
    Y = np.array(Y)
    M = np.array(M)
    return X, Y, M


def split_data_by_time(seq, flag=None):
    length = len(seq)
    split_train_time = int(length * 0.9)
    split_val_time = int(length * 1.0)

    seq_train = seq[:split_train_time]
    seq_val = seq[split_train_time - (flag * 24 * 6):split_val_time]
    seq_test = seq[split_val_time - (flag * 24 * 6):]

    return seq_train, seq_val, seq_test


def generate_train_val_test_for_npz(data, masks, npz_path, lag, horizon, save=False):
    print('Load WP Dataset shaped: ', data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # split dataset by days or by ratio
    data_with_mask = np.concatenate([data, np.expand_dims(masks, axis=-1)], axis=2)
    data_train, data_val, data_test = split_data_by_time(data_with_mask, 2)

    # add time window
    step_size = 6  # TODO
    x_train, y_train, m_train = Add_Window_Horizon(data_train, step_size=step_size)
    x_val, y_val, m_val = Add_Window_Horizon(data_val, step_size=step_size)
    x_test, y_test, m_test = Add_Window_Horizon(data_test, step_size=step_size)

    if save:
        for cat in ["train", "val", "test"]:
            _x, _y, _m = locals()["x_" + cat], locals()["y_" + cat], locals()["m_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape, "m:", _m.shape)
            np.savez_compressed(
                os.path.join(npz_path, f"{cat}.npz"),
                x=_x,
                y=_y,
                m=_m,
            )
    return x_train, y_train, m_train, x_val, y_val, m_val, x_test, y_test, m_test


def preprocess_data(data_path, npz_path, horizon, lags):
    if not os.path.exists(npz_path):
        os.mkdir(npz_path)

    data = {}
    if 'train.npz' not in os.listdir(npz_path):
        print("Running data preprocessing")
        all_data = TestData(data_path)
        seqs, _, masks = all_data.get_all_turbines()
        seqs = np.stack(seqs, axis=1)
        masks = np.stack(masks, axis=1)
        data_tuple = generate_train_val_test_for_npz(seqs, masks, npz_path, lag=lags, horizon=horizon, save=True)
        x_train, y_train, m_train, x_val, y_val, m_val, x_test, y_test, m_test = data_tuple

        data['x_train'] = x_train
        data['y_train'] = y_train
        data['m_train'] = m_train
        data['x_val'] = x_val
        data['y_val'] = y_val
        data['m_val'] = m_val
        data['x_test'] = x_test
        data['y_test'] = y_test
        data['m_test'] = m_test
    else:
        print("Data already preprocessed")
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(npz_path, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y'][:, :, :, -1:] if cat_data['x'].shape[0] > 0 else cat_data['y']
            data['m_' + category] = cat_data['m']
    return data

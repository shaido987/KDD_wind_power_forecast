import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class TestData(object):
    def __init__(self, path_to_data, task='MS', target='Patv', start_col=3, farm_capacity=134):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        self.df_data = deepcopy(self.df_raw)
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


def add_window_horizon(data, window=3, horizon=1, single=False, flag=6):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + flag
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def add_window_horizon_with_mask(data, window=3, horizon=1, single=False, flag=6):
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
    M = []      # mask
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + window, :, :-1])
            Y.append(data[index + window + horizon - 1:index + window + horizon, :, :-1])
            M.append(data[index + window + horizon - 1:index + window + horizon, :, -1:])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window, :, :-1])
            Y.append(data[index + window:index + window + horizon, :, :-1])
            M.append(data[index + window:index + window + horizon, :, -1:])
            index = index + flag
    X = np.array(X)
    Y = np.array(Y)
    M = np.array(M)
    return X, Y, M


def split_data_by_time(df, save_path, flag=None, train_ratio=None, mask=False):
    ratio = train_ratio / 10
    length = len(df)
    split_train_time = int(length * ratio)

    df_train = df[:split_train_time]
    df_val = df[split_train_time - (flag * 24 * 6):]
    df_test = df[split_train_time - (flag * 24 * 6):]

    if mask:
        feature_dims = df_train.shape[-1] - 1
    else:
        feature_dims = df_train.shape[-1]
    scaler = StandardScaler().fit(df_train[:, :, :feature_dims].reshape(-1, feature_dims))

    print("Saving scaler")
    scaler_path = os.path.join(save_path, 'scaler.pickle')
    with open(scaler_path, 'wb') as handle:
        pickle.dump(scaler, handle, protocol=4)

    return df_train, df_val, df_test


def generate_train_val_test_for_npz(save_path, data, lag, horizon):
    print('Load WP Dataset shaped: ', data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # split dataset by days or by ratio
    flag = 2
    ratio = 9
    data_train, _, data_test = split_data_by_time(data, save_path, flag, train_ratio=ratio)

    # add time window
    x_train, y_train = add_window_horizon(data_train, lag, horizon, single=False, flag=flag)
    x_test, y_test = add_window_horizon(data_test, lag, horizon, single=False, flag=1)

    for cat in ["train", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez(os.path.join(save_path, f"{cat}_{ratio}_new_2.npz"), x=_x, y=_y)


def generate_train_val_test_mask_for_npz(save_path, data, mask, lag, horizon, ratio):
    print('Load WP Dataset shaped: ', data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # split dataset by days or by ratio
    flag = 2
    data_with_mask = np.concatenate([data, np.expand_dims(mask, axis=-1)], axis=2)
    data_train, _, data_test = split_data_by_time(data_with_mask, save_path, flag, train_ratio=ratio, mask=True)

    # add time window
    x_train, y_train, m_train = add_window_horizon_with_mask(data_train, lag, horizon, single=False, flag=flag)
    x_test, y_test, m_test = add_window_horizon_with_mask(data_test, lag, horizon, single=False, flag=1)

    for cat in ["train", "test"]:
        _x, _y, _m = locals()["x_" + cat], locals()["y_" + cat], locals()["m_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez(os.path.join(save_path, f"{cat}_{ratio}_new_mask.npz"), x=_x, y=_y, m=_m)


def run_data_preprocess(input_data, save_path, lag, horizon, ratio):
    data = TestData(input_data)
    seqs, _, mask = data.get_all_turbines()
    seqs = np.stack(seqs, axis=1)
    masks = np.stack(mask, axis=1)
    generate_train_val_test_mask_for_npz(save_path, seqs, masks, lag=lag, horizon=horizon, ratio=ratio)

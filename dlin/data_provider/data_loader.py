import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from dlin.utils.feature_engineering import construct_features

warnings.filterwarnings('ignore')


def parse_data(df):
    df['Day'] = pd.to_datetime([d - 1 for d in df['Day'].values], unit='D', origin=pd.Timestamp('2022-01-01'))
    df['date'] = pd.to_datetime(df['Day'].astype(str) + ' ' + df['Tmstamp'])
    df = df.drop(columns=['Day', 'Tmstamp'])
    df = df.set_index('date')

    # Add missing timestamps
    start = df.index.min()
    end = df.index.max()
    timestamps = pd.date_range(start=start, end=end, freq='10Min')
    df = df.groupby('TurbID', group_keys=False).apply(lambda x: x.reindex(timestamps))
    df = df.rename_axis('date')

    # Remove incorrect or unknown values
    nan_rows = df.isnull().any(axis=1)
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5)) | \
                   ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)) | \
                   ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
                    (df['Ndir'] > 720))
    df.loc[invalid_cond, set(df.columns) - {'TurbID'}] = np.nan
    df['mask'] = np.where(invalid_cond | nan_rows, 0, 1)

    # Handle power features less than 0
    df.loc[df['Patv'] < 0, 'Patv'] = 0
    df['Prtv'] = df['Prtv'].abs()

    df = df.groupby('TurbID').apply(lambda x: x.interpolate().ffill().bfill().fillna(0))

    df = df.reset_index()
    df = construct_features(df)
    return df


class Dataset_Custom(Dataset):
    def __init__(self, data_path, flag='train', size=None, target='Patv', scale=False, **kwargs):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        df_raw = parse_data(df_raw)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        train_split = int(len(df_raw) * 0.9)
        border1s = [0, train_split]
        border2s = [train_split, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        n_turbine = 134
        df_raw = df_raw.sort_values('date').iloc[border1:border2].sort_values(['TurbID', 'date'])

        drop_cols = ['TurbID', 'date', 'mask']
        if self.scale:
            train_data = df_raw.sort_values('date').iloc[border1s[0]:border2s[0]].sort_values(['TurbID', 'date'])

            self.scaler.fit(train_data.drop(columns=drop_cols).values)
            data = self.scaler.transform(df_raw.drop(columns=drop_cols).values)
        else:
            data = df_raw.drop(columns=drop_cols).values

        self.data_x = data.reshape(n_turbine, -1, data.shape[-1])
        self.data_y = data.reshape(n_turbine, -1, data.shape[-1])
        self.mask = df_raw['mask'].values.reshape(n_turbine, -1, 1)

    def __getitem__(self, index):
        turbine = index // (self.data_x.shape[1] - self.seq_len - self.pred_len + 1)
        s_begin = index % (self.data_x.shape[1] - self.seq_len - self.pred_len + 1)
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[turbine][s_begin:s_end]
        seq_y = self.data_y[turbine][r_begin:r_end]
        mask = self.mask[turbine][r_begin:r_end]
        return seq_x, seq_y, mask

    def __len__(self):
        return self.data_x.shape[0] * (self.data_x.shape[1] - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        if data.shape[-1] == 1 and self.scaler.n_features_in_ > 1:
            self.scaler.n_features_in_ = 1
            self.scaler.mean_ = self.scaler.mean_[-1:]
            self.scaler.scale_ = self.scaler.scale_[-1:]
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, data_path, size=None, target='Patv', scale=False, scaler=None, **kwargs):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        df_raw = parse_data(df_raw)
        self.raw_data = df_raw.copy()

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        cols_data = df_raw.columns[1:].drop('mask')
        df_data = df_raw[cols_data]

        data = df_data.groupby('TurbID').apply(lambda x: x[-self.seq_len:])
        values = data.drop(columns='TurbID').reset_index(drop=True).values.reshape(134, self.seq_len, -1)

        if self.scale:
            values = self.scaler.transform(values.reshape(-1, values.shape[2])).reshape(134, self.seq_len, -1)

        self.data_x = values

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        return seq_x

    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        if data.shape[-1] == 1 and self.scaler.n_features_in_ > 1:
            self.scaler.n_features_in_ = 1
            self.scaler.mean_ = self.scaler.mean_[-1:]
            self.scaler.scale_ = self.scaler.scale_[-1:]
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataLoader(object):
    def __init__(self, xs, ys, ms, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            m_padding = np.repeat(ms[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ms = np.concatenate([ys, m_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ms = ms

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, ms = self.xs[permutation], self.ys[permutation], self.ms[permutation]
        self.xs = xs
        self.ys = ys
        self.ms = ms

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                m_i = self.ms[start_ind: end_ind, ...]
                yield (x_i, y_i, m_i)
                self.current_ind += 1
        return _wrapper()


def sample_ids(ids, k):
    """
    sample `k - 1` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    # sampled_ids.append(ids[-1])  # must sample the centroid node itself
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids


def cos_dis(X):
    """
    cosine distance
    :param X: (batch_size, N, d)
    :return: (batch_size, N, N)
    """
    X = nn.functional.normalize(X, dim=2)
    XT = X.transpose(1, 2)
    return torch.matmul(X, XT)


def cos_dis_version_2(X):
    """
    cosine distance
    :param X: (batch_size, N, d)
    :return: (batch_size, N, N)
    """
    X = nn.functional.normalize(X, dim=2)
    X = X.transpose(0, 1).contiguous().view(X.shape[1], -1)
    XT = X.transpose(0, 1)
    return torch.matmul(X, XT)


def nearest_select(feats, kn):
    """

    :param feats: [batch_size, dim, num_nodes, channel]
    :param kn:
    :return: idx: [batch_size, num_nodes, idx_nearest_nodes]
    """
    feats = feats.transpose(1,2)
    feats = feats.contiguous().view(feats.size(0), feats.size(1), -1)

    dis = cos_dis(feats)
    _, idx = torch.topk(dis, kn, dim=2)
    return idx


def _generate_G_from_wavenet(output, kn):
    """
    :param output: list of features [batch_size, channels, num_nodes, n_dim]
    :param kn:
    :return: [batch_size, num_nodes, n_neighbor, n_dim]
    """
    hyperedges = []
    for feats in output:
        idx = nearest_select(feats, kn)
        hyperedges.append(idx)
    edges = torch.cat(hyperedges, dim=2)
    batch_size, num_nodes, n_neighbor = edges.size()
    edges = edges.contiguous().view(-1, edges.size(-1))
    # x = output[-1].transpose(1,2).squeeze()
    x = output[-1].transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
    n_dim = x.size(-1)
    x = x.contiguous().view(-1, x.size(-1))
    x = x[edges.view(-1)].view(-1, n_neighbor, n_dim)
    x = x.view(batch_size, num_nodes, n_neighbor, n_dim)
    return x


def load_wp_dataset(data, batch_size, valid_batch_size=None, test_batch_size=None):
    # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    _, seq_x, num_nodes, _ = data['x_train'].shape

    feature_dims = data['x_train'].shape[3]
    scaler = StandardScaler().fit(data['x_train'].reshape(-1, feature_dims))
    target_scaler = StandardScaler()
    target_scaler.mean_ = scaler.mean_[-1:]
    target_scaler.scale_ = scaler.scale_[-1:]

    # Data format, We only normalized the data for x
    for category in ['train', 'val', 'test']:
        # if test set is empty
        if category == 'test' and data['x_test'].shape[0] == 0:
            continue

        x_col, y_col = 'x_' + category, 'y_' + category
        data[x_col] = scaler.transform(data[x_col].reshape(-1, feature_dims)).reshape(data[x_col].shape)

        # TODO: Don't scale the y values?
        data[y_col] = target_scaler.transform(data[y_col].reshape(-1, 1)).reshape(data[y_col].shape)

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], data['m_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], data['m_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], data['m_test'], test_batch_size)

    data['scaler'] = scaler
    return data, num_nodes

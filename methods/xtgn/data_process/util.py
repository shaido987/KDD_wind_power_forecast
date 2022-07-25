import os
import torch
import numpy as np


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
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
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self, flag=None):
        self.current_ind = 0
        if flag == 'Train':
            self.sample_batch_num = self.num_batch // 2
        else:
            self.sample_batch_num = self.num_batch

        def _wrapper():
            while self.current_ind < self.sample_batch_num:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


class DataLoader_mask(object):
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
            ms = np.concatenate([ms, m_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ms = ms

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]
        self.ms = self.ms[permutation]

    def get_iterator(self, flag=None):
        self.current_ind = 0
        if flag == 'Train':
            self.sample_batch_num = self.num_batch // 2
        else:
            self.sample_batch_num = self.num_batch

        def _wrapper():
            while self.current_ind < self.sample_batch_num:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                m_i = self.ms[start_ind: end_ind, ...]
                yield (x_i, y_i, m_i)
                self.current_ind += 1
        return _wrapper()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=2, verbose=False, delta=0, model_save_path='best_model_{}.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 2
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
            model_save_path (str): Model save path and model name.
                            Dafault: None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_save = model_save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_save)
        self.val_loss_min = val_loss


def load_wp_dataset_mask(dataset_dir, ratio, batch_size, test_batch_size=None, scaler=None):
    data = {}
    for category in ['train', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + f"_{ratio}_new_mask.npz"))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][:, :, :, -1:]
        data['m_' + category] = cat_data['m']

    _, seq_x, num_nodes, _ = data['x_train'].shape
    feature_dims = data['x_train'].shape[3]

    # Data format, We only normalized the data for x
    for category in ['train', 'test']:
        if category == 'test' and data['x_test'].shape[0] == 0:
            continue
        x_col, y_col = 'x_' + category, 'y_' + category
        data[x_col] = scaler.transform(data[x_col].reshape(-1, feature_dims)).reshape(data[x_col].shape)

    data['train_loader'] = DataLoader_mask(data['x_train'], data['y_train'], data['m_train'], batch_size)
    data['test_loader'] = DataLoader_mask(data['x_test'], data['y_test'], data['m_test'], test_batch_size)
    data['scaler'] = scaler
    return data, num_nodes

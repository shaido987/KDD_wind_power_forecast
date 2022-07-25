import torch
import numpy as np
import torch.nn as nn


def adjust_learning_rate(optimizer, epoch, args):
    if args['lradj'] == 'type1':
        lr_adjust = {epoch: args['learning_rate'] * (0.5 ** ((epoch - 1) // 1))}
    elif args['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args['lradj'] == '3':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 10 else args['learning_rate']*0.1}
    elif args['lradj'] == '4':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 15 else args['learning_rate']*0.1}
    elif args['lradj'] == '5':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 25 else args['learning_rate']*0.1}
    elif args['lradj'] == '6':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 5 else args['learning_rate']*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, name='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, path, 'checkpoint-last.pth')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + name)
        self.val_loss_min = val_loss


class CustomMaskedLoss(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true, mask):
        mask = mask[:, -self.pred_len:, :]
        pred = pred[:, -self.pred_len:, -1:] * mask
        true = true[:, -self.pred_len:, -1:] * mask

        loss = torch.sqrt(self.mse(pred, true)) + self.mae(pred, true)
        return loss
import torch
import numpy as np


def masked_mae(preds, labels, mask, null_val=np.nan):
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(loss)


def masked_mse(preds, labels, mask, null_val=np.nan):
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.mean(loss)


def masked_rmse(preds, labels, mask, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, mask, null_val))


def masked_mape(preds, labels, mask, null_val=np.nan):
    # mask /= torch.mean(mask)
    loss = torch.abs(preds - labels) / labels
    loss *= mask
    return torch.mean(loss)


def metric(preds, labels, mask, null_val=np.nan):
    mae = masked_mae(preds, labels, mask, null_val)
    mape = masked_mape(preds, labels, mask, null_val)
    rmse = masked_rmse(preds, labels, mask, null_val)
    return mae, mape, rmse


def mape(preds, labels, null_val=np.nan):
    mask = labels != null_val
    weight = mask.float()
    weight = mask / torch.mean((weight))
    loss = torch.abs(preds - labels - 1) / (labels + 1)
    loss = loss[mask] * weight[mask]
    return torch.mean(loss)

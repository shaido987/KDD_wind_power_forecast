import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


def cos_dis(X):
    """
    cosine distance
    :param X: (batch_size, N, d)
    :return: (batch_size, N, N)
    """
    X = nn.functional.normalize(X, dim=1)
    XT = X.transpose(0, 1)
    return torch.matmul(X, XT)


def compute_adj_matrix(input_data, save_path):
    adj_matrix = pd.read_csv(input_data, index_col=0).to_numpy()
    adj_matrix = torch.tensor(adj_matrix)
    sim = cos_dis(adj_matrix)
    num = 50
    _, idx = torch.topk(sim, num, dim=1)

    b = 1 / num
    c = (1 - b) / (num - 1)
    a = np.zeros([134, 134])
    for i in range(134):
        a[i, idx[i]] = c

    for i in range(134):
        a[i, i] = b

    np.savetxt(save_path, a, delimiter=",")

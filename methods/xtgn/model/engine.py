import torch
import torch.nn.functional as F

from methods.xtgn.model.MVSTM_1 import MVSTDM
from methods.xtgn.data_process.metrics import *


class trainer():
    def __init__(self, device, scaler, num_nodes, seq_length_x, in_dim, out_dim, seq_length_y, weight_decay,
                 dropout_rate, milestones, num_epochs, print_freq, batch_size, residual_channels, gamma, clip,
                 dilation_channels, skip_channels, end_channels, blocks, layers, kernel_size, learning_rate, embed_dim,
                 adj_matrix):

        self.model = MVSTDM(
            in_dim=in_dim,
            num_nodes=num_nodes,
            seq_length_x=seq_length_x,
            seq_length_y=seq_length_y,
            dropout_rate=dropout_rate,
            out_dim=out_dim,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            blocks=blocks,
            layers=layers,
            kernel_size=kernel_size,
            batch_size=batch_size,
            embed_dim=embed_dim,
            adj_matrix=adj_matrix
        )

        self.milestones = [int(i) for i in milestones]
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = masked_mae
        self.scaler = scaler
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        self.num_nodes = num_nodes
        self.device = device

    def train(self, input, real_val, mask=None, ite=None):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input, flag='Train')
        # output = self.scaler.inverse_transform(output)

        loss = self.loss(output, real_val, mask, 0.0)
        loss.backward()
        self.optimizer.step()

        mape = masked_mape(output, real_val, 0.0).item()
        rmse = masked_rmse(output, real_val, mask, 0.0).item()

        return loss.item(), mape, rmse

    def eval(self, input, real_val, mask=None, ite=None):
        self.model.eval()
        output = self.model(input, flag='Val')

        loss = self.loss(output, real_val, mask, 0.0)

        mape = masked_mape(output, real_val, 0.0).item()
        rmse = masked_rmse(output, real_val, mask, 0.0).item()

        return loss.item(), mape, rmse

    def test(self, input, mask=None):
        self.model.eval()
        output = self.model(input, flag='Test')
        output = F.relu(output)
        return output

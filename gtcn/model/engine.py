import torch

from gtcn.model.MVSTM_1 import MVSTDM
from gtcn.data_process.metrics import *


class trainer():
    def __init__(self, device, scaler, num_nodes, seq_length_x, in_dim, out_dim, seq_length_y, weight_decay,
                 dropout_rate, milestones, gamma, num_epochs, print_freq, clip, batch_size, residual_channels,
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
        # """signal-layer"""
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.model.spatial_temporal_prediction[0].parameters(), 'lr': transformer_learning_rate},
        #     {'params': self.model.spatial_temporal_prediction[1].parameters()}
        # ], lr=hgnn_learning_rate, weight_decay=weight_decay, eps=1.0e-8)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=gamma)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = clip
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        self.num_nodes = num_nodes
        self.device = device

        self.loss2 = torch.nn.SmoothL1Loss()
        self.loss3 = masked_rmse

    def train(self, input, real_val, mask, ite):
        """
        :param ite:
        :param input:
        :param real_val:
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()

        # input = input.to(self.device)
        output = self.model(input)
        # output = self.scaler.inverse_transform(output)

        loss = self.loss(output, real_val, mask, 0.0)
        """ train_3 测试中 """
        # loss = self.loss2(predict, real_val)
        """ train_4测试中 """
        # loss = 0.5 * self.loss(predict, real_val, 0.0) + 0.5 * self.loss3(predict, real_val, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = masked_mape(output, real_val, mask, 0.0).item()
        rmse = masked_rmse(output, real_val, mask, 0.0).item()

        return loss.item(), mape, rmse

    def eval(self, input, real_val, mask, ite):
        self.model.eval()
        # input = input.to(self.device)
        output = self.model(input)
        # output = self.scaler.inverse_transform(output)
        loss = self.loss(output, real_val, mask, 0.0)
        """ train_3 测试中 """
        # loss = self.loss2(predict, real_val)
        """ train_4测试中 """
        # loss = 0.5 * self.loss(predict, real_val, 0.0) + 0.5 * self.loss3(predict, real_val, 0.0)

        mape = masked_mape(output, real_val, mask, 0.0).item()
        rmse = masked_rmse(output, real_val, mask, 0.0).item()

        return loss.item(), mape, rmse

    def test(self, input):
        self.model.eval()
        output = self.model(input)
        return output

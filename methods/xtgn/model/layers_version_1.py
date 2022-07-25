import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.xtgn.data_process.util import *


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_nodes, embed_dim, adj_matrix):
        super(GCN, self).__init__()
        self.liner = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = dropout
        self.embedding = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.independent_conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=True)
        self.independent_conv_2 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=True)

        self.shared_conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=True)
        self.shared_conv_2 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=True)

        self.adj_matrix = adj_matrix

    def forward(self, x, flag="Train"):
        x = self.liner(x)
        """Location adj_matrix"""
        if flag == "Test":
            x = torch.matmul(self.adj_matrix, x)

        return x

class WaveNet(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, out_dim, residual_channels, dilation_channels, skip_channels,
                 end_channels, blocks, layers, kernel_size, batch_size):
        super(WaveNet, self).__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        receptive_filed = 0

        """ WaveNet. """
        for i in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for j in range(self.layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_filed += additional_scope
                additional_scope *= 2

        self.receptive_filed = receptive_filed
        self.end_conv = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
                                  kernel_size=(1, 288 - self.receptive_filed), bias=True)
        self.attention_conv = nn.Conv2d(in_channels=self.blocks + 1, out_channels=1,
                                        kernel_size=(1, 1), bias=True)

    def forward(self, input):
        in_len = input.size(3)
        output = []
        if in_len < self.receptive_filed:
            x = nn.functional.pad(input, (self.receptive_filed - in_len, 0, 0, 0))
        else:
            x = input
        output.append(x)
        x = self.start_conv(x)
        skip = 0

        """ WaveNet Layers"""
        for i in range(self.blocks):
            for j in range(self.layers):
                #            |----------------------------------------|     *residual*
                #            |                                        |
                #            |    |-- conv -- tanh --|                |
                # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
                #                 |-- conv -- sigm --|     |
                #                                         1x1
                #                                          |
                # ---------------------------------------> + ------------->	*skip*
                count = i * 2 + j
                residual = x
                # dilated convolution
                filter = self.filter_convs[count](residual)
                filter = torch.tanh(filter)
                gate = self.gate_convs[count](residual)
                gate = torch.sigmoid(gate)
                x = filter * gate

                # parametrized skip connection
                s = x
                s = self.skip_convs[count](s)
                try:
                    skip = skip[:, :, :, -s.size(3):]
                except:
                    skip = 0
                skip = s + skip
                # if count % 2 == 1:
                #     output.append(s)

                x = self.residual_convs[count](x)
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[count](x)
            # every block
            output.append(s)


        x = skip
        x = self.end_conv(x)
        x = x.squeeze(-1).transpose(1, 2)

        return x

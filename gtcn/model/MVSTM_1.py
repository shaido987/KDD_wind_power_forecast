import torch.nn as nn

from gtcn.model.layers_version_1 import WaveNet, GCN


class MVSTDM(nn.Module):
    def __init__(self, in_dim, num_nodes, seq_length_x, seq_length_y, dropout_rate, out_dim, residual_channels,
                 dilation_channels, skip_channels, end_channels, blocks, layers, kernel_size, batch_size, embed_dim,
                 adj_matrix):
        """
        :param in_dim:
        :param num_nodes:
        :param seq_length_x:
        :param seq_length_y:
        :param dropout_rate:
        :param out_dim:
        :param residual_channels:
        :param dilation_channels:
        :param skip_channels:
        :param end_channels:
        :param blocks:
        :param layers:
        :param batch_size:
        """
        super(MVSTDM, self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.seq_length_x = seq_length_x
        self.seq_length_y = seq_length_y
        self.dropout_rate = dropout_rate

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        self.ops = nn.ModuleList(
            [WaveNet(
                num_nodes=self.num_nodes,
                dropout=self.dropout_rate,
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                residual_channels=self.residual_channels,
                dilation_channels=self.dilation_channels,
                skip_channels=self.skip_channels,
                end_channels=self.end_channels,
                blocks=self.blocks,
                layers=self.layers,
                kernel_size=self.kernel_size,
                batch_size=self.batch_size
            )] + [GCN(
                in_dim=self.end_channels,
                out_dim=self.out_dim,
                dropout=self.dropout_rate,
                num_nodes=self.num_nodes,
                embed_dim=self.embed_dim,
                adj_matrix=adj_matrix
            )])

        self.linear = nn.Linear(self.end_channels, 288, bias=True)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        x, adj = self.ops[0](input)
        # x = self.linear(x)
        x = self.ops[1](x=x, A=adj)

        return x
"""
Note:
    This is a reconstitution version from  pytorch_geometric/ torch_geometric / nn / models / schnet.py.
"""
from math import pi as PI

import torch
import torch.nn.functional as F
from ase.data import atomic_masses
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch_geometric.nn import radius_graph, MessagePassing, Set2Set
from torch_scatter import scatter


"""This is one general script. For different data, you should re-write this and tune."""

import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Sequential, GRU
from torch_geometric.nn import NNConv, Set2Set

from models.models_geo.basemodel import BaseCrystalModel, ShiftedSoftplus


class SchNet(BaseCrystalModel):
    def __init__(self, **kwargs):
        super(SchNet, self).__init__(**kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = ModuleList()
        for _ in range(self.num_interactions):
            block = SchNet_InteractionBlock(self.hidden_channels, self.num_gaussians, self.num_filters, self.cutoff)
            self.interactions.append(block)


class SchNet_InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(SchNet_InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = _CFConv(hidden_channels, hidden_channels, num_filters,
                            self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, data=None):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class _CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(_CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)  # W 传递给message, 这里不是必须的
        x = self.lin2(x)
        return x

    def message(self, x_j, W):  # [num_edge,]
        return x_j * W

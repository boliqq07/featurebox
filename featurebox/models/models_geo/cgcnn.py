"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F

from torch.nn import Module, ModuleList, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv

from models.models_geo.basemodel import BaseCrystalModel

class CGCNN_Interactions(Module):
    """auto attention."""
    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2,
                 ):
        super(CGCNN_Interactions, self).__init__()
        nf = num_filters
        self.lin0 = Linear(hidden_channels, nf)
        self.short = Linear(num_gaussians, 3)

        nn = Sequential(Linear(3, hidden_channels), ReLU(), Linear(hidden_channels, nf * nf))

        self.conv = NNConv(nf, nf, nn, aggr='mean')
        self.n_conv = n_conv

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.relu(self.lin0(h))
        edge_attr = F.relu(self.short(edge_attr))

        for i in range(self.n_conv):
            out = F.relu(self.conv(out, edge_index, edge_attr))

        return out


class CrystalGraphConvNet(BaseCrystalModel):
    def __init__(self,*args, num_gaussians=5, num_filters=64, hidden_channels=64, **kwargs):
        super(CrystalGraphConvNet, self).__init__(*args,num_gaussians=num_gaussians, num_filters=num_filters, hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None # not used for this network.

    def get_interactions_layer(self):
        self.interactions = CGCNN_Interactions(self.hidden_channels,self.num_gaussians, self.num_filters,
                                                n_conv=self.num_interactions,)
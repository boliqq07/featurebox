"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Sequential, GRU
from torch_geometric.nn import NNConv, Set2Set

from models.models_geo.basemodel import BaseCrystalModel


class CGGRU_Interactions(Module):
    """auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2,
                 ):
        super(CGGRU_Interactions, self).__init__()
        nf = num_filters
        self.lin0 = Linear(hidden_channels, nf)

        self.short = Linear(num_gaussians, 3)

        nn = Sequential(Linear(3, hidden_channels), ReLU(), Linear(hidden_channels, nf * nf))

        self.conv = NNConv(nf, nf, nn, aggr='mean')
        self.n_conv = n_conv
        self.gru = GRU(nf, nf)

    def forward(self, h, edge_index, edge_weight, edge_attr, **kwargs):
        out = F.relu(self.lin0(h))

        edge_attr = F.relu(self.short(edge_attr))
        h = out.unsqueeze(0)

        for i in range(self.n_conv):
            m = F.relu(self.conv(out, edge_index, edge_attr))

            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

            # import pynvml
            # pynvml.nvmlInit()
            # handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # 0表示显卡标号
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print(meminfo.free / 1024 ** 2)  # 剩余显存大小

        return out


class CGGRU_ReadOut(Module):
    def __init__(self, num_filters=128,
                 n_set2set=2):
        super(CGGRU_ReadOut, self).__init__()
        nf = num_filters
        self.set2set = Set2Set(nf, processing_steps=n_set2set)  # very import
        self.lin1 = Linear(2 * nf, nf)
        self.lin2 = Linear(nf, 1)

    def forward(self, out, batch):
        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class CGGRUNet(BaseCrystalModel):
    def __init__(self,*args,num_gaussians=5, num_filters=64, hidden_channels=64,**kwargs):
        super(CGGRUNet, self).__init__(*args, num_filters=num_filters,
                                       num_gaussians=num_gaussians,
                                       hidden_channels= hidden_channels,  **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = CGGRU_Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                                n_conv=self.num_interactions,)

    def get_readout_layer(self):
        self.readout_layer = CGGRU_ReadOut(self.num_filters,
                                           n_set2set=2)

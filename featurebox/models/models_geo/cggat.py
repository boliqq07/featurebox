"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
from torch_geometric.nn import GATConv

from featurebox.models.models_geo.basemodel import BaseCrystalModel


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2,
                 ):
        super(_Interactions, self).__init__()
        _ = num_gaussians
        self.lin0 = Linear(hidden_channels, num_filters)

        self.conv = ModuleList()

        for _ in range(n_conv):
            nn = GATConv(
                in_channels=num_filters, out_channels=num_filters,
                add_self_loops=False,
                bias=True, )
            self.conv.append(nn)

        self.n_conv = n_conv

    def forward(self, x, edge_index, edge_weight, edge_attr, **kwargs):
        out = F.relu(self.lin0(x))

        for convi in self.conv:
            out = out+ F.relu(convi(x=out, edge_index=edge_index))

        return out


class CrystalGraphGAT(BaseCrystalModel):
    """
    CrystalGraph with GAT.
    """

    def __init__(self, *args, num_gaussians=5, num_filters=64, hidden_channels=64, **kwargs):
        super(CrystalGraphGAT, self).__init__(*args, num_gaussians=num_gaussians, num_filters=num_filters,
                                              hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, )

"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Module, ModuleList
from torch_geometric.nn import CGConv

from featurebox.models.models_geo.basemodel import BaseCrystalModel
from featurebox.utils.general import collect_edge_attr_jump, lift_jump_index_select


class CGConvNew(CGConv):
    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64,
                 n_conv=2, jump=True, ):
        super(_Interactions, self).__init__()

        _ = jump
        short_len = 1  # not gauss
        assert short_len == 1, "keep 1 for sparse tensor"

        self.lin0 = Linear(hidden_channels, num_filters)
        self.short = Linear(num_gaussians, short_len)
        self.conv = ModuleList()

        for _ in range(n_conv):
            nn = CGConvNew(channels=num_filters, dim=short_len,
                           aggr='add', batch_norm=False,
                           bias=True, )
            self.conv.append(nn)

        self.n_conv = n_conv

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.relu(self.lin0(h))
        edge_attr = F.relu(self.short(edge_attr))
        # edge_attr = edge_attr.reshape(-1, 1)
        for convi in self.conv:
            out = out + F.relu(convi(x=out, edge_index=edge_index, edge_attr=edge_attr))

        return out


class CrystalGraphConvNet(BaseCrystalModel):
    """
    CrystalGraph.
    """

    def __init__(self, *args, num_gaussians=5, num_filters=64, hidden_channels=64, **kwargs):
        super(CrystalGraphConvNet, self).__init__(*args, num_gaussians=num_gaussians, num_filters=num_filters,
                                                  hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, jump=self.jump)

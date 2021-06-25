"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Linear
from torch.nn import Module, ModuleList
from torch_geometric.data.sampler import Adj
from torch_geometric.nn import CGConv

from featurebox.models.models_geo.basemodel import BaseCrystalModel
from featurebox.utils.general import temp_jump, temp_jump_cpu, check_device


class CGConvJump(CGConv):
    """# torch.geometric scatter is unstable especially for small data in cuda device.!!!"""

    @property
    def device(self):
        return check_device(self)

    @temp_jump_cpu()
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        return super(CGConvJump, self).propagate(edge_index, size=size, **kwargs)

    @temp_jump()
    def message(self, x_i, x_j, edge_attr) -> Tensor:
        return super(CGConvJump, self).message(x_i, x_j, edge_attr)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2,
                 ):
        super(_Interactions, self).__init__()
        self.lin0 = Linear(hidden_channels, num_filters)
        short_len = 20
        self.short = Linear(num_gaussians, short_len)

        self.conv = ModuleList()

        for _ in range(n_conv):
            nn = CGConvJump(channels=num_filters, dim=short_len,
                            aggr='add', batch_norm=True,
                            bias=True, )
            self.conv.append(nn)

        self.n_conv = n_conv

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.relu(self.lin0(h))
        edge_attr = F.relu(self.short(edge_attr))

        for convi in self.conv:
            out = out + F.relu(convi(x=out, edge_index=edge_index, edge_attr=edge_attr))

        return out


class CrystalGraphConvNet(BaseCrystalModel):
    """
    CrystalGraph with GAT.
    """

    def __init__(self, *args, num_gaussians=5, num_filters=64, hidden_channels=64, **kwargs):
        super(CrystalGraphConvNet, self).__init__(*args, num_gaussians=num_gaussians, num_filters=num_filters,
                                                  hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, )

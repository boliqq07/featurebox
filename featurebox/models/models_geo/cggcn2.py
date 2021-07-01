"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
from torch_geometric.nn import GCN2Conv

from featurebox.models.models_geo.basemodel import BaseCrystalModel
# class GCN2ConvJump(GCN2Conv):
#     """# torch.geometric scatter is unstable especially for small data in cuda device.!!!"""
#
#     @property
#     def device(self):
#         return check_device(self)
#
#     @temp_jump_cpu()
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         return super().propagate(edge_index, size=size, **kwargs)
#
#     @temp_jump()
#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return super().message(x_j, edge_weight)
from featurebox.utils.general import collect_edge_attr_jump, lift_jump_index_select


class GCNConv2New(GCN2Conv):
    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """
    Auto attention.
    """

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2, jump=True,
                 ):
        super(_Interactions, self).__init__()
        _ = num_gaussians
        self.lin0 = Linear(hidden_channels, num_filters)

        self.conv = ModuleList()
        for _ in range(n_conv):
            nn = GCNConv2New(
                channels=num_filters, alpha=0.9, theta=None,
                layer=None, shared_weights=True,
                cached=False, add_self_loops=False, normalize=True, )
            self.conv.append(nn)
        self.n_conv = n_conv

    def forward(self, x, edge_index, edge_weight, edge_attr, **kwargs):
        x = F.relu(self.lin0(x))
        out = x
        for convi in self.conv:
            out = out + F.relu(convi(x=out, x_0=x, edge_index=edge_index, edge_weight=edge_weight))
        return out


class CrystalGraphGCN2(BaseCrystalModel):
    """CrystalGraph with GCN2."""

    def __init__(self, *args, num_gaussians=50, num_filters=128, hidden_channels=128, **kwargs):
        super(CrystalGraphGCN2, self).__init__(*args, num_gaussians=num_gaussians, num_filters=num_filters,
                                               hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, jump=self.jump)

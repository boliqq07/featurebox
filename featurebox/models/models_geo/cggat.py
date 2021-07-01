"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Module, ModuleList
from torch_geometric.nn import CGConv, GATConv

from featurebox.models.models_geo.basemodel import BaseCrystalModel
from featurebox.utils.general import collect_edge_attr_jump, lift_jump_index_select


# class GATConvJump(GATConv):
#     """# torch.geometric scatter is unstable especially for small data in cuda device.!!!"""
#
#     @property
#     def device(self):
#         return check_device(self)
#
#     @temp_jump_cpu()
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         return super(GATConvJump, self).propagate(edge_index, size=size, **kwargs)
#
#     @temp_jump()
#     def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#         return super(GATConvJump, self).message(x_j, alpha_j, alpha_i,
#                                                 index, ptr,
#                                                 size_i)


class GATConvNew(GATConv):
    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2, jump=True,
                 ):
        super(_Interactions, self).__init__()
        _ = jump
        _ = num_gaussians

        self.lin0 = Linear(hidden_channels, num_filters)
        self.conv = ModuleList()

        for _ in range(n_conv):
            nn = GATConvNew(
                in_channels=num_filters, out_channels=num_filters,
                add_self_loops=False,
                bias=True, )
            self.conv.append(nn)

        self.n_conv = n_conv

    def forward(self, x, edge_index, edge_weight, edge_attr, **kwargs):

        out = F.relu(self.lin0(x))
        for convi in self.conv:
            out = out + F.relu(convi(x=out, edge_index=edge_index))

        return out


class CrystalGraphGAT(BaseCrystalModel):
    """
    CrystalGraph with GAT.
    """

    def __init__(self, *args, num_gaussians=5, num_filters=32, hidden_channels=64, **kwargs):
        super(CrystalGraphGAT, self).__init__(*args, num_gaussians=num_gaussians, num_filters=num_filters,
                                              hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, jump=self.jump)

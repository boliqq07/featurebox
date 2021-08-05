"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

from typing import Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import Module, ModuleList
from torch_geometric.nn import CGConv
from torch_geometric.typing import PairTensor, OptTensor, Adj, Size

from featurebox.models_geo.basemodel import BaseCrystalModel
from featurebox.models_geo.general import collect_edge_attr_jump, lift_jump_index_select


class CGConvNew(CGConv):

    def __init__(self, *args, **kwargs):
        super(CGConvNew, self).__init__(*args, **kwargs)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out = out + x[1]
        return out

    def __collect__(self, args, edge_index, size, kwargs):
        return collect_edge_attr_jump(self, args, edge_index, size, kwargs)

    def __lift__(self, src, edge_index, dim):
        return lift_jump_index_select(self, src, edge_index, dim)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, node_hidden_channels=64, num_edge_gaussians=None, num_node_interaction_channels=64,
                 n_conv=1, **kwargs):
        super(_Interactions, self).__init__()

        short_len = 1  # not gauss
        assert short_len == 1, "keep 1 for sparse tensor"

        self.lin0 = Linear(node_hidden_channels, num_node_interaction_channels)
        self.short = Linear(num_edge_gaussians, short_len)
        self.conv = ModuleList()

        for _ in range(n_conv):
            cg = CGConvNew(channels=num_node_interaction_channels, dim=short_len,
                           aggr='add', batch_norm=True,
                           bias=True, )
            self.conv.append(cg)

        self.n_conv = n_conv

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.short.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, h, edge_index, edge_weight, edge_attr, data):
        out = F.softplus(self.lin0(h))
        edge_attr = F.softplus(self.short(edge_attr))

        for convi in self.conv:
            out = out + convi(x=out, edge_index=edge_index, edge_attr=edge_attr)

        return out

class CrystalGraphConvNet(BaseCrystalModel):
    """
    CrystalGraph.
    """

    def __init__(self, *args, num_edge_gaussians=None, num_node_interaction_channels=16, num_node_hidden_channels=8,
                 **kwargs):
        super(CrystalGraphConvNet, self).__init__(*args, num_edge_gaussians=num_edge_gaussians,
                                                  num_node_interaction_channels=num_node_interaction_channels,
                                                  num_node_hidden_channels=num_node_hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.num_node_hidden_channels, self.num_edge_gaussians,
                                          self.num_node_interaction_channels,
                                          n_conv=self.num_interactions,
                                          kwargs=self.interaction_kwargs)

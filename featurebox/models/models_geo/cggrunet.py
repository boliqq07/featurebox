"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Linear
from torch.nn import ReLU, Sequential, GRU
from torch_geometric.data.sampler import Adj
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from featurebox.models.models_geo.basemodel import BaseCrystalModel
from featurebox.utils.general import check_device, temp_jump_cpu
from featurebox.utils.general import temp_jump


class NNConvJump(NNConv):
    """# torch.geometric scatter is unstable especially for small data in cuda device.!!!"""

    @property
    def device(self):
        return check_device(self)

    @temp_jump_cpu()
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        return super().propagate(edge_index, size=size, **kwargs)

    @temp_jump()
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return super().message(x_j, edge_attr)


class Set2SetJump(Set2Set):

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = self.tem_scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    @temp_jump_cpu()
    def tem_scatter_add(self, x, batch, dim, dim_size):
        return scatter_add(x, batch, dim=dim, dim_size=dim_size)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class _Interactions(Module):
    """Auto attention."""

    def __init__(self, hidden_channels=64, num_gaussians=5, num_filters=64, n_conv=2, jump=True,
                 ):
        super(_Interactions, self).__init__()
        nf = num_filters
        self.lin0 = Linear(hidden_channels, nf)

        self.short = Linear(num_gaussians, nf)

        nn = Sequential(Linear(nf, nf // 4), ReLU(), Linear(nf // 4, nf * nf))
        if jump:
            self.conv = NNConvJump(nf, nf, nn, aggr='mean')
        else:
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

        return out


class CGGRU_ReadOut(Module):
    def __init__(self, num_filters=128, jump=True,
                 n_set2set=2, out_size=1):
        super(CGGRU_ReadOut, self).__init__()
        nf = num_filters
        if jump:
            self.set2set = Set2SetJump(nf, processing_steps=n_set2set)  # very import
        else:
            self.set2set = Set2Set(nf, processing_steps=n_set2set)  # very import
        self.lin1 = Linear(2 * nf, nf)
        self.lin2 = Linear(nf, out_size)

    def forward(self, out, batch):
        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


class CGGRUNet(BaseCrystalModel):
    """
    CrystalGraph with CGGRUN.
    """

    def __init__(self, *args, num_gaussians=5, num_filters=64, hidden_channels=64, **kwargs):
        super(CGGRUNet, self).__init__(*args, num_filters=num_filters,
                                       num_gaussians=num_gaussians,
                                       hidden_channels=hidden_channels, **kwargs)
        self.num_state_features = None  # not used for this network.

    def get_interactions_layer(self):
        self.interactions = _Interactions(self.hidden_channels, self.num_gaussians, self.num_filters,
                                          n_conv=self.num_interactions, jump=self.jump)

    def get_readout_layer(self):
        self.readout_layer = CGGRU_ReadOut(self.num_filters,
                                           n_set2set=2, out_size=self.out_size, jump=self.jump)

import warnings

import torch
from torch_cluster import radius_graph
from torch_geometric.utils import remove_self_loops


class GaussianSmearing:
    """Smear the radius shape (num_node,1) to shape (num_node, num_edge_gaussians)."""

    def __init__(self, start=0.0, stop=5.0, num_edge_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.offset = torch.linspace(start, stop, num_edge_gaussians)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]).item() ** 2

    def __call__(self, data):
        dist = data.edge_weight
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        if hasattr(data, "edge_attr") and data.edge_attr.shape[1] != 1:
            warnings.warn("The old edge_attr is covered by smearing edge_weight", UserWarning)
        data.edge_attr = torch.exp(self.coeff * torch.pow(dist, 2))

        return data


class AddEdge:
    """For (BaseStructureGraphGEO) without edge index, this is one lazy way to calculate edge ."""

    def __init__(self, cutoff=7.0, ):
        self.cutoff = cutoff

    def __call__(self, data):

        edge_index = radius_graph(data.pos, r=self.cutoff, batch=data.batch)
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index[0], edge_index[1]
        edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
        if hasattr(data, "edge_attr"):
            pass
        else:
            data.edge_attr = edge_weight.reshape(-1, 1)
        return data


class NormalizeStateAttr(object):

    def __init__(self):
        self.scale = None

    def __call__(self, data):
        if self.scale is None:
            self.scale = data.state_attr.sum(1, keepdim=True)
        data.state_attr = data.state_attr / self.scale.clamp(min=1)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeX(object):

    def __init__(self):
        self.scale = None

    def __call__(self, data):
        if self.scale is None:
            self.scale = data.x.sum(1, keepdim=True)
        data.x = data.x / self.scale.clamp(min=1)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeEdgeAttr(object):

    def __init__(self):
        self.scale = None

    def __call__(self, data):
        if self.scale is None:
            self.scale = data.edge_attr.sum(1, keepdim=True)
        data.edge_attr = data.edge_attr / self.scale.clamp(min=1)

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class DistributionEdgeAttr:
    def __init__(self, r_index_num=0, r_cs=0.3, r_c=6.0):
        super().__init__()
        self.r_index_num = r_index_num
        self.r_cs = r_cs
        self.r_c = r_c

    def __call__(self, data):
        x = data.edge_attr
        r_mark = x[:, self.r_index_num]
        r_m1 = r_mark < self.r_cs
        r_m3 = r_mark > self.r_c
        r_m2 = ~(r_m3 & r_m1)
        x[r_m1] = 1 / x[r_m1]
        x[r_m2] = 1 / x[r_m2] * (0.5 * torch.cos(torch.pi * (x[r_m2] - self.r_cs) / (self.r_c - self.r_cs)) + 0.5)
        x[r_m3] = 0
        data.edge_attr=x
        return data
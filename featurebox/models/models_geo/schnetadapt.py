"""
Note:
    This is a reconstitution version from  pytorch_geometric/ torch_geometric / nn / models / schnet.py.
"""
from math import pi as PI

import torch
import torch.nn.functional as F
from ase.data import atomic_masses
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch_geometric.nn import radius_graph, MessagePassing, Set2Set
from torch_scatter import scatter


class SchNetAdapt(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self, num_node_features=1, num_bond_features=3, num_embeddings=100,
                 hidden_channels=128, num_filters=128,
                 num_interactions=3, num_gaussians=50, cutoff=10.0,
                 readout='add', dipole=False, mean=None, std=None,
                 atomref=None, simple_z=True, simple_edge=True, ):
        super(SchNetAdapt, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.simple_edge = simple_edge
        self.simple_z = simple_z

        atomic_mass = torch.from_numpy(atomic_masses)  # 嵌入原子质量
        self.register_buffer('atomic_mass', atomic_mass)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。
        if simple_z:
            assert num_node_features == 0, "simple_z just accept num_node_features == 0"
            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(2, hidden_channels) # not used
        else:
            assert num_node_features > 0, "simple_z just accept num_node_features == 0"
            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(num_node_features, hidden_channels)
        if simple_edge:
            assert num_bond_features == 0, "The `num_node_features` must be the same size with `x` feature."
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        else:
            assert num_bond_features > 0, "The `num_bond_features` must be the same size with `edge_attr` feature."
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians - num_bond_features)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        # self.set2set = Set2Set(hidden_channels, processing_steps=2)  # very import

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)
            # 单个原子的性质是否加到最后

        self.reset_parameters()
        if self.simple_edge == True:
            print(
                "simple_edge == True means re-calculate and arrange the edge_index,edge_weight,edge_attr."
                "The old would be neglected")

    def reset_parameters(self):
        self.embedding_e.reset_parameters()
        self.embedding_l.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, data):
        # 使用embedding 作为假的原子特征输入，而没有用原子特征输入
        assert hasattr(data, "z")
        assert hasattr(data, "pos")
        assert hasattr(data, "batch")
        z = data.z
        pos = data.pos
        batch = data.batch
        # 处理数据阶段
        if self.simple_z:
            # 处理数据阶段
            assert z.dim() == 1 and z.dtype == torch.long
            batch = torch.zeros_like(z) if batch is None else batch
            h = self.embedding_e(z)
        else:
            assert hasattr(data, "x")
            x = data.x
            h1 = self.embedding_e(z)
            h2 = self.embedding_l(x)
            h = h1+h2

        if self.simple_edge:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
        else:
            assert hasattr(data, "edge_attr")
            assert hasattr(data, "edge_index")
            if not hasattr(data, "edge_weight"):
                row, col = data.edge_index
                edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            else:
                edge_weight = data.edge_weight
            edge_attr = self.distance_expansion(edge_weight)
            edge_index = data.edge_index
            edge_attr = torch.cat((data.edge_attr, edge_attr), dim=1)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # 加入偶极矩
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out.view(-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)  # W 传递给message, 这里不是必须的
        x = self.lin2(x)
        return x

    def message(self, x_j, W):  # [num_edge,]
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

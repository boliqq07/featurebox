"""This part contains some base model and tools which just for crystal problem."""
import ase.data as ase_data
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential, ReLU
from torch.nn import Module
from torch_geometric.nn import radius_graph
from torch_scatter import scatter


class GaussianSmearing(Module):
    """Smear the radius."""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(Module):
    """Softplus with one log2 intercept."""
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift




class BaseCrystalModel(Module):
    """Base model for crystal problem."""
    def __init__(self,
                 num_node_features=1,
                 num_bond_features=3,
                 num_state_features=0,
                 num_embeddings=100,
                 hidden_channels=128,
                 num_filters=128,
                 num_interactions=3,
                 num_gaussians=50,
                 cutoff=10.0,
                 readout='add',
                 dipole=False,
                 mean=None,
                 std=None,
                 atomref=None,
                 simple_z=True,
                 simple_edge=True,
                 interactions=None,
                 readout_layer=None,
                 add_state=False,
                 ):
        """

        Args:
            num_node_features: (int) number of node feature (atom feature).
            num_bond_features: (int) number of bond feature.
            num_state_features: (int) number of state feature.
            num_embeddings: (int) number of embeddings, For generate the initial embedding matrix
                to on behalf of node feature.
            hidden_channels: (int)
            num_filters: (int)
            num_interactions: (int)
            num_gaussians: (int) number of gaussian Smearing number for radius.
            cutoff: (float) cutoff for calculate neighbor bond, just used for ``simple_edge`` == True.
            readout: (str) Merge node method. such as "add","mean","max","mean".
            dipole: (bool) dipole.
            mean: (float) mean
            std: (float) std
            atomref: (torch.tensor shape (120,1)) properties for atom. such as target y is volumes of compound,
                atomref could be the atom volumes of all atom (H,H,He,Li,...). And you could copy the first term to
                make sure the `H` index start form 1.
            simple_z: (bool) just used "z" or used "x" to calculate.
            simple_edge: (bool) True means re-calculate and arrange the edge_index, edge_weight, edge_attr.
                The old would be neglected.
            interactions: (Callable) torch module for interactions.
                dynamic: pass the torch module to interactions parameter.
                static: re-define the ``get_interactions_layer`` and keep this parameter is None.
                the forward input is (h, edge_index, edge_weight, edge_attr, data=data)
            readout_layer: (Callable) torch module for interactions.
                dynamic: pass the torch module to interactions parameter.
                static: re-define the ``get_interactions_layer`` and keep this parameter is None.
                the forward input is (out,)
            add_state: (bool) add state attribute before output.
        """
        super(BaseCrystalModel, self).__init__()

        # 初始定义
        assert readout in ['add', 'sum', 'mean']
        self.hidden_channels = hidden_channels
        self.num_state_features = num_state_features
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
        self.interactions = interactions
        self.readout_layer = readout_layer
        # 嵌入原子属性，备用
        # (嵌入别太多，容易慢，大多数情况下用不到。)
        atomic_mass = torch.from_numpy(ase_data.atomic_masses)  # 嵌入原子质量
        covalent_radii = torch.from_numpy(ase_data.covalent_radii)  # 嵌入共价半径
        self.register_buffer('atomic_mass', atomic_mass)
        self.register_buffer('atomic_mass', covalent_radii)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。

        # 定义输入
        # 使用原子性质,或者使用Embedding 产生随机数据。
        # 使用键性质,或者使用Embedding 产生随机数据。

        if simple_z:
            assert num_node_features == 0, "simple_z just accept num_node_features == 0, and don't " \
                                           "use your self-defined 'x' data"
            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(2, hidden_channels)  # not used
        else:
            assert num_node_features > 0, "The `num_node_features` must be the same size with `x` feature."
            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(num_node_features, hidden_channels)
        if simple_edge:
            print(
                "simple_edge == True means re-calculate and arrange the edge_index,edge_weight,edge_attr."
                "The old would be neglected")
            assert num_bond_features == 0, "The `num_node_features` must be the same size with `edge_attr` feature." \
                                           "and don't use your self-defined 'edge_attr' data"
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        else:
            assert num_bond_features > 0, "The `num_bond_features` must be the same size with `edge_attr` feature."
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians - num_bond_features)

        # 交互层 需要自定义
        if interactions is None:
            self.get_interactions_layer()
        # 合并层 需要自定义
        if readout_layer is None:
            self.get_readout_layer()

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(120, 1)
            self.atomref.weight.data.copy_(atomref)
            # 单个原子的性质是否加到最后

        self.add_state=add_state
        if self.add_state:
            self.ads = Linear(self.hidden_channels+self.num_state_features,self.hidden_channels)
            self.adsrelu = ReLU(inplace=True)
        self.reset_parameters()

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
            h = h1 + h2

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

        if isinstance(self.interactions, ModuleList):
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr, data=data)
        else:
            h = self.interactions(h, edge_index, edge_weight, edge_attr, data=data)

        out = self.readout_layer(h, batch)

        if self.add_state:
            assert hasattr(data, "state_attr"), "the ``add_state`` must accpet ``state_attr`` in data."
            out = self.ads(torch.cat((out, data.state_attr), dim=1))
            out = self.adsrelu(out)

        return self.output_forward(out)

    def get_interactions_layer(self):
        """This part shloud re-defined. And must add the ``interactions`` attribute.

        Examples::

            >>> self.interactions = torch.nn.ModuleList(...)

        Examples::

            >>> ...
            >>> self.interactions = YourNet()
        """
        self.interactions = ModuleList()
        for _ in range(self.num_interactions):
            block = self.InteractionBlock(self.hidden_channels, self.num_gaussians, self.num_filters, self.cutoff)
            self.interactions.append(block)

    def get_readout_layer(self):
        """This part shloud re-defined. And must add the ``readout_layer`` attribute.

        Examples::

            >>> self.readout_layer = torch.nn.Sequential(...)

        Examples::

            >>> ...
            >>> self.readout_layer = YourNet()
        """
        self.readout_layer = ReadOutLayer(readout=self.readout)

    def output_forward(self, out):

        if not self.dipole and self.mean is not None and self.std is not None:
            out = out * self.std + self.mean

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out.view(-1)

    def internal_forward(self, h, z, pos, batch, *args, **kwargs):

        if self.dipole:
            # 加入偶极矩
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')

    def reset_parameters(self):
        self.embedding_e.reset_parameters()
        self.embedding_l.reset_parameters()
        if isinstance(self.interactions,ModuleList):
            for interaction in self.interactions:
                interaction.reset_parameters()
        else:
            self.interactions.reset_parameters()

        self.readout_layer.reset_parameters()
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)


class ReadOutLayer(Module):
    """Merge node"""

    def __init__(self, readout="add"):
        super(ReadOutLayer, self).__init__()
        self.readout = readout
        self.lin = Sequential(
            Linear(self.hidden_channels, self.hidden_channels // 2),
            ShiftedSoftplus(),
            Linear(self.hidden_channels // 2, 1), )

    def forward(self, h, batch):
        h = self.lin(h)
        return scatter(h, batch, dim=0, reduce=self.readout)

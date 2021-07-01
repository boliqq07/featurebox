"""This part contains base model for crystal problem and tools."""
import warnings

import ase.data as ase_data
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, LayerNorm
from torch.nn import Module
from torch_scatter import segment_csr

from featurebox.utils.general import get_ptr


class ShiftedSoftplus(Module):
    """Softplus with one log2 intercept."""

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class BaseCrystalModel(Module):
    """
    Base model for crystal problem.
    """

    def __init__(self,
                 num_node_features=1,
                 num_bond_features=3,
                 num_state_features=0,
                 num_embeddings=120,
                 hidden_channels=128,
                 num_filters=128,
                 num_interactions=1,
                 num_gaussians=50,
                 cutoff=10.0,
                 out_size=1,
                 readout='add',
                 dipole=False,
                 mean=None,
                 std=None,
                 atomref=None,
                 simple_z=True,
                 interactions=None,
                 readout_layer=None,
                 add_state=False,
                 jump=False,
                 ):
        """
        Args:
            num_node_features: (int) input number of node feature (atom feature).
            num_bond_features: (int) input number of bond feature, wich is equal to .
            if ``num_gaussians`` offered, this parameter is neglect.
            num_state_features: (int) input number of state feature.
            num_embeddings: (int) number of embeddings, For generate the initial embedding matrix
            to on behalf of node feature.
            hidden_channels: (int) hidden_channels for node feature.
            num_filters: (int) num_filters for node feature.
            num_interactions: (int) conv number.
            num_gaussians: (int) number of gaussian Smearing number for radius.
            keep this compact with your bond data.
            cutoff: (float) deprecated! cutoff for calculate neighbor bond, deprecated!
            jump: (float) jump cpu to cal scatter. deprecated!
            readout: (str) Merge node method. such as "add","mean","max","mean".
            dipole: (bool) dipole.
            mean: (float) mean
            std: (float) std
            atomref: (torch.tensor shape (120,1)) properties for atom. such as target y is volumes of compound,
            atomref could be the atom volumes of all atom (H,H,He,Li,...). And you could copy the first term to
            make sure the `H` index start form 1.
            simple_z: (bool) just used "z" or used "x" to calculate.
            interactions: (Callable) torch module for interactions.dynamic: pass the torch module to interactions parameter.static: re-define the ``get_interactions_layer`` and keep this parameter is None.
            the forward input is (h, edge_index, edge_weight, edge_attr, data=data)
            readout_layer: (Callable) torch module for interactions.dynamic: pass the torch module to interactions parameter. static: re-define the ``get_interactions_layer`` and keep this parameter is None. the forward input is (out,)
            add_state: (bool) add state attribute before output.
            out_size:(int) number of out size. for regression,is 1 and for classification should be defined.
        """
        super(BaseCrystalModel, self).__init__()

        # 初始定义
        if num_gaussians is None:
            num_gaussians = num_bond_features
        assert readout in ['add', 'sum', 'min', 'mean', "max", 'mul']
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
        self.simple_z = simple_z
        self.interactions = interactions
        self.readout_layer = readout_layer
        self.out_size = out_size
        self.jump = jump

        # 嵌入原子属性，备用
        # (嵌入别太多，容易慢，大多数情况下用不到。)
        atomic_mass = torch.from_numpy(ase_data.atomic_masses)  # 嵌入原子质量
        covalent_radii = torch.from_numpy(ase_data.covalent_radii)  # 嵌入共价半径
        self.register_buffer('atomic_mass', atomic_mass)
        self.register_buffer('atomic_radii', covalent_radii)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。

        # 定义输入
        # 使用原子性质,或者使用Embedding 产生随机数据。
        # 使用键性质,或者使用Embedding 产生随机数据。
        if num_embeddings < 120:
            print("default, num_embeddings>=120,if you want simple the net work and "
                  "This network does not apply to other elements, the num_embeddings could be less but large than "
                  "the element type number in your data.")

        # 原子个数，一般不用动，这是所有原子种类数，
        # 一般来说，采用embedding的网络，
        # 在向其他元素（训练集中没有的）数据推广的能力较差。

        if simple_z is True:
            if num_node_features != 0:
                warnings.warn("simple_z just accept num_node_features == 0, "
                              "and don't use your self-defined 'x' data, but element number Z", UserWarning)

            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(2, 2)  # not used
            self.embedding_l2 = Linear(2, 2)  # not used
        elif self.simple_z == "no_embed":
            self.embedding_e = Linear(2, 2)
            self.embedding_l = Linear(num_node_features, hidden_channels)
            self.embedding_l2 = Linear(hidden_channels, hidden_channels)
        else:
            assert num_node_features > 0, "The `num_node_features` must be the same size with `x` feature."
            self.embedding_e = Embedding(num_embeddings, hidden_channels)
            self.embedding_l = Linear(num_node_features, hidden_channels)
            self.embedding_l2 = Linear(hidden_channels, hidden_channels)

        # 交互层 需要自定义
        if interactions is None:
            self.get_interactions_layer()
        # 合并层 需要自定义
        if readout_layer is None:
            self.get_readout_layer()

        self.register_buffer('initial_atomref', atomref)
        if atomref is None:
            self.atomref = atomref
        elif isinstance(atomref, Tensor) and atomref.shape[0] == 120:
            self.atomref = lambda x: atomref[x]
        elif atomref == "Embed":
            self.atomref = Embedding(120, 1)
            self.atomref.weight.data.copy_(atomref)
            # 单个原子的性质是否加到最后
        else:
            self.atomref = atomref

        self.add_state = add_state
        if self.add_state:
            self.dp = LayerNorm(self.num_state_features)
            self.ads = Linear(self.num_state_features, 1)
            self.ads2 = Linear(1, self.out_size)

        self.reset_parameters()

    def forward(self, data):
        # 使用embedding 作为假的原子特征输入，而没有用原子特征输入
        assert hasattr(data, "z")
        assert hasattr(data, "pos")
        assert hasattr(data, "batch")
        z = data.z
        batch = data.batch
        pos = data.pos
        # 处理数据阶段
        if self.simple_z is True:
            # 处理数据阶段
            assert z.dim() == 1 and z.dtype == torch.long
            # batch = torch.zeros_like(z) if batch is None else batch
            h = self.embedding_e(z)
            h = F.relu(h)
        elif self.simple_z == "no_embed":
            assert hasattr(data, "x")
            x = data.x
            h = F.relu(self.embedding_l(x))
            h = self.embedding_l2(h)
        else:
            assert hasattr(data, "x")
            x = data.x
            h1 = self.embedding_e(z)
            x = F.relu(self.embedding_l(x))
            h2 = self.embedding_l2(x)
            h = h1 + h2

        data.x = h

        if data.edge_index is None:
            edge_index = data.adj_t
        else:
            edge_index = data.edge_index

        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise NotImplementedError("Must offer edge_weight or edge_attr.")
            else:
                if data.edge_attr.shape[1] == 1:
                    data.edge_weight = data.edge_attr.reshape(-1)
                else:
                    data.edge_weight = data.edge_attr
        # 自定义
        if isinstance(self.interactions, ModuleList):
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, data.edge_weight, data.edge_attr,
                                    data=data)
        else:
            h = self.interactions(h, edge_index, data.edge_weight, data.edge_attr,
                                  data=data)

        h = self.internal_forward(h, z, pos, batch)

        # 自定义
        out = self.readout_layer(h, batch)

        if self.add_state:
            assert hasattr(data, "state_attr"), "the ``add_state`` must accept ``state_attr`` in data."
            sta = self.dp(data.state_attr)
            sta = self.ads(sta)

            sta = F.relu(sta)

            out = self.ads2(out.expand_as(sta) + sta)

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
        self.readout_layer = ReadOutLayer(num_filters=self.num_filters, readout=self.readout, out_size=self.out_size)

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

            c = segment_csr(mass * pos, get_ptr(batch)) / segment_csr(mass, get_ptr(batch))
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
        self.embedding_l2.reset_parameters()
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)


class ReadOutLayer(Module):
    """Merge node layer."""

    def __init__(self, num_filters, out_size=1, readout="add", ):
        super(ReadOutLayer, self).__init__()
        self.readout = readout
        self.lin1 = Linear(num_filters, num_filters * 5)
        self.s1 = ShiftedSoftplus()
        self.lin2 = Linear(num_filters * 5, num_filters)
        self.s2 = ShiftedSoftplus()
        self.lin3 = Linear(num_filters, out_size)

    def forward(self, h, batch):
        h = self.lin1(h)
        h = self.s1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        h = self.lin2(h)
        h = self.s2(h)
        h = self.lin3(h)
        return h

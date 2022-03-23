"""This part contains base model for crystal problem and tools."""
import warnings
from abc import abstractmethod

import ase.data as ase_data
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, LayerNorm, ModuleList, Softplus, ReLU, Sequential, BatchNorm1d
from torch.nn import Module
from torch_geometric.nn import radius_graph
from torch_geometric.utils import remove_self_loops
from torch_scatter import segment_csr

from featurebox.models_geo.general import get_ptr


class BaseCrystalModel(Module):
    """
    Base model for crystal problem.
    """

    def __init__(self,
                 num_node_features=1,
                 num_edge_features=3,
                 num_state_features=0,

                 num_node_hidden_channels=128,
                 num_node_interaction_channels=128,
                 num_interactions=1,
                 num_edge_gaussians=None,

                 num_node_embeddings=120,
                 cutoff=10.0,
                 out_size=1,
                 readout='add',
                 mean=None,
                 std=None,
                 norm=False,
                 atom_ref=None,
                 simple_z=True,
                 interactions=None,
                 readout_layer=None,
                 add_state=False,
                 **kwargs
                 ):
        """
        Model for crystal problem.

        Args:
            num_node_features: (int) input number of node feature (atom feature).
            num_edge_features: (int) input number of bond feature. if ``num_edge_gaussians` offered, this parameter is neglect.
            num_state_features: (int) input number of state feature.
            num_node_embeddings: (int) number of embeddings, For generate the initial embedding matrix to on behalf of node feature.
            num_node_hidden_channels: (int) num_node_hidden_channels for node feature.
            num_node_interaction_channels: (int) channels for node feature.
            num_interactions: (int) conv number.
            num_edge_gaussians: (int) number of gaussian Smearing number for radius. deprecated, keep this compact with your bond data.
            cutoff: (float) cutoff for calculate neighbor bond.
            readout: (str) Merge node method. such as "add","mean","max","mean".
            mean: (float) mean for y.
            std: (float) std for y.
            norm:(bool) False or True norm for y.
            atom_ref: (torch.tensor shape (120,1)) properties for atom. such as target y is volumes of compound,
                atom_ref could be the atom volumes of all atom (H,H,He,Li,...). And you could copy the first term to
                make sure the `H` index start form 1.
            simple_z: (bool,str) just used "z" or used "x" to calculate.
            interactions: (Callable) torch module for interactions dynamically: pass the torch module to interactions parameter.static: re-define the ``get_interactions_layer`` and keep this parameter is None.
                the forward input is (h, edge_index, edge_weight, edge_attr, data=data)
            readout_layer: (Callable) torch module for interactions  dynamically: pass the torch module to interactions parameter. static: re-define the ``get_interactions_layer`` and keep this parameter is None. the forward input is (out,)
            add_state: (bool) add state attribute before output.
            out_size:(int) number of out size. for regression,is 1 and for classification should be defined.
        """
        super(BaseCrystalModel, self).__init__()

        self.interaction_kwargs = {}
        for k, v in kwargs.items():
            if "interaction_kwargs_" in k:
                self.interaction_kwargs[k.replace("interaction_kwargs_", "")] = v

        self.readout_kwargs = {}
        for k, v in kwargs.items():
            if "readout_kwargs_" in k:
                self.readout_kwargs[k.replace("readout_kwargs_", "")] = v

        # 初始定义
        if num_edge_gaussians is None:
            num_edge_gaussians = num_edge_features

        assert readout in ['add', 'sum', 'min', 'mean', "max"]

        self.num_node_hidden_channels = num_node_hidden_channels
        self.num_state_features = num_state_features
        self.num_node_interaction_channels = num_node_interaction_channels
        self.num_interactions = num_interactions
        self.num_edge_gaussians = num_edge_gaussians
        self.cutoff = cutoff
        self.readout = readout

        self.mean = mean
        self.std = std
        self.scale = None
        self.simple_z = simple_z
        self.interactions = interactions
        self.readout_layer = readout_layer
        self.out_size = out_size
        self.norm = norm

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
        if num_node_embeddings < 120:
            print("default, num_node_embeddings>=120,if you want simple the net work and "
                  "This network does not apply to other elements, the num_node_embeddings could be less but large than "
                  "the element type number in your data.")

        # 原子个数，一般不用动，这是所有原子种类数，
        # 一般来说，采用embedding的网络，
        # 在向其他元素（训练集中没有的）数据推广的能力较差。

        if simple_z is True:
            if num_node_features != 0:
                warnings.warn("simple_z just accept num_node_features == 0, "
                              "and don't use your self-defined 'x' data, but element number Z", UserWarning)

            self.embedding_e = Embedding(num_node_embeddings, num_node_hidden_channels)
            # self.embedding_l = Linear(2, 2)  # not used
            # self.embedding_l2 = Linear(2, 2)  # not used
        elif self.simple_z == "no_embed":
            # self.embedding_e = Linear(2, 2)
            self.embedding_l = Linear(num_node_features, num_node_hidden_channels)
            self.embedding_l2 = Linear(num_node_hidden_channels, num_node_hidden_channels)
        else:
            assert num_node_features > 0, "The `num_node_features` must be the same size with `x` feature."
            self.embedding_e = Embedding(num_node_embeddings, num_node_hidden_channels)
            self.embedding_l = Linear(num_node_features, num_node_hidden_channels)
            self.embedding_l2 = Linear(num_node_hidden_channels, num_node_hidden_channels)

        self.bn = BatchNorm1d(num_node_hidden_channels)

        # 交互层 需要自定义 get_interactions_layer
        if interactions is None:
            self.get_interactions_layer()
        elif isinstance(interactions, ModuleList):
            self.get_res_interactions_layer(interactions)
        elif isinstance(interactions, Module):
            self.interactions = interactions
        else:
            raise NotImplementedError("please implement get_interactions_layer function, "
                                      "or pass interactions parameters.")
        # 合并层 需要自定义
        if readout_layer is None:
            self.get_readout_layer()
        elif isinstance(readout_layer, Module):
            self.readout_layer = readout_layer
        else:
            raise NotImplementedError("please implement get_readout_layer function, "
                                      "or pass readout_layer parameters.")

        self.register_buffer('initial_atom_ref', atom_ref)

        if atom_ref is None:
            self.atom_ref = atom_ref
        elif isinstance(atom_ref, Tensor) and atom_ref.shape[0] == 120:
            self.atom_ref = lambda x: atom_ref[x]
        elif atom_ref == "embed":
            self.atom_ref = Embedding(120, 1)
            self.atom_ref.weight.data.copy_(atom_ref)
            # 单个原子的性质是否加到最后
        else:
            self.atom_ref = atom_ref

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
        # pos = data.pos
        # 处理数据阶段
        if self.simple_z is True:
            # 处理数据阶段
            assert z.dim() == 1 and z.dtype == torch.long

            h = self.embedding_e(z)
            h = F.softplus(h)
        elif self.simple_z == "no_embed":
            assert hasattr(data, "x")
            x = data.x
            h = F.softplus(self.embedding_l(x))
            h = self.embedding_l2(h)
        else:
            assert hasattr(data, "x")
            x = data.x
            h1 = self.embedding_e(z)
            x = F.softplus(self.embedding_l(x))
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
                    data.edge_weight = torch.norm(data.edge_attr, dim=1, keepdim=True)

        h = self.bn(h)

        h = self.interactions(h, edge_index, data.edge_weight, data.edge_attr,
                              data=data)

        h = self.internal_forward(h, z)

        out = self.readout_layer(h, batch)

        if self.add_state:
            assert hasattr(data, "state_attr"), "the ``add_state`` must accept ``state_attr`` in data."
            sta = self.dp(data.state_attr)
            sta = self.ads(sta)
            sta = F.relu(sta)
            out = self.ads2(out.expand_as(sta) + sta)

        return self.output_forward(out)

    def get_res_interactions_layer(self, interactions):
        self.interactions = GeoResNet(interactions)

    @abstractmethod
    def get_interactions_layer(self):
        """This part shloud re-defined. And must add the ``interactions`` attribute.

        Examples::

            >>> ...
            >>> self.interactions = YourNet()
        """

    def get_readout_layer(self):
        """This part shloud re-defined. And must add the ``readout_layer`` attribute.

        Examples::

            >>> self.readout_layer = torch.nn.Sequential(...)

        Examples::

            >>> ...
            >>> self.readout_layer = YourNet()
        """
        if "readout_kwargs_layers_size" in self.readout_kwargs:
            self.readout_layer = GeneralReadOutLayer(**self.readout_kwargs)
        else:
            self.readout_layer = GeneralReadOutLayer(
                [self.num_node_interaction_channels, self.readout, 200, self.out_size], **self.readout_kwargs)

    def output_forward(self, out):

        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean

        if self.norm is True:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out.view(-1)

    def dipole_forward(self, h, z, pos, batch):
        # 加入偶极矩
        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = segment_csr(mass * pos, get_ptr(batch)) / segment_csr(mass, get_ptr(batch))
        h = h * (pos - c[batch])

        return h

    def internal_forward(self, h, z):
        if self.atom_ref is not None:
            h = h + self.atom_ref(z)

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_node_hidden_channels={self.num_node_hidden_channels}, '
                f'num_node_interaction_channels={self.num_node_interaction_channels}, '
                f'num_interactions={self.num_interactions}, '
                f'num_edge_gaussians={self.num_edge_gaussians}, '
                f'cutoff={self.cutoff})')

    def reset_parameters(self):
        if hasattr(self, "embedding_e"):
            self.embedding_e.reset_parameters()
        if hasattr(self, "embedding_l"):
            self.embedding_l.reset_parameters()
        if hasattr(self, "embedding_l2"):
            self.embedding_l2.reset_parameters()
        self.bn.reset_parameters()
        if self.atom_ref is not None:
            self.atom_ref.weight.data.copy_(self.initial_atomref)


class ReadOutLayer(Module):
    """Merge node layer."""

    def __init__(self, channels, out_size=1, readout="add", ):
        super(ReadOutLayer, self).__init__()
        self.readout = readout
        self.lin1 = Linear(channels, channels * 10)
        self.s1 = ShiftedSoftplus()
        self.lin2 = Linear(channels * 10, channels * 5)
        self.s2 = ShiftedSoftplus()
        self.lin3 = Linear(channels * 5, out_size)

    def forward(self, h, batch):
        h = self.lin1(h)
        h = self.s1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        h = self.lin2(h)
        h = self.s2(h)
        h = self.lin3(h)
        return h


class GeneralReadOutLayer(Module):
    """General Merge node layer.
    """

    def __init__(self, layers_size=(128, "sum", 32, 1), last_layer=None, active_layer_type="ShiftedSoftplus"):
        super(GeneralReadOutLayer, self).__init__()
        l = len(layers_size)
        readout = [i for i in layers_size if isinstance(i, str)]
        if active_layer_type == "ShiftedSoftplus":
            active_layer = ShiftedSoftplus
        elif active_layer_type == "Softplus":
            active_layer = Softplus
        elif active_layer_type == "ReLU":
            active_layer = ReLU
        elif isinstance(active_layer_type, Module):
            active_layer = active_layer_type.__class__
        else:
            raise NotImplementedError("can't identify the type of layer.")

        assert len(readout) == 1, "The readout layer must be set one time, please there are {} layer: {}.".format(
            len(readout), readout)

        readout_site = [n for n, i in enumerate(layers_size) if isinstance(i, str)][0]
        readout = layers_size[readout_site]
        assert readout in ('sum', 'max', 'min', 'mean', 'add')
        self.readout = readout
        layers_size = list(layers_size)
        layers_size[readout_site] = layers_size[readout_site - 1]

        part_first = []
        part_second = []
        i = 0
        while i < l - 1:
            if i < readout_site - 1:
                part_first.append(Linear(layers_size[i], layers_size[i + 1]))
                part_first.append(active_layer())
            elif i < readout_site:
                pass
            else:
                part_second.append(Linear(layers_size[i], layers_size[i + 1]))
                part_second.append(active_layer())
            i += 1

        self.modules_list_1 = Sequential(*part_first)
        self.modules_list_2 = Sequential(*part_second)
        self.last_layer = last_layer

    def reset_parameters(self):
        self.modules_list_1.reset_parameters()
        self.modules_list_2.reset_parameters()
        self.last_layer.reset_parameters()

    def forward(self, h, batch):
        if len(self.modules_list_1) > 0:
            h = self.modules_list_1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        if len(self.modules_list_2) > 0:
            h = self.modules_list_2(h)
        if self.last_layer is not None:
            h = self.last_layer(h)
        return h


class GeoResNet(Module):
    """Simple ResNet"""

    def __init__(self, module_list: ModuleList):
        super().__init__()
        self.modules_list = module_list

    def forward(self, h, edge_index, edge_weight, edge_attr, data=None):
        for interaction in self.modules_list:
            h = h + interaction(h, edge_index, edge_weight, edge_attr, data=data)
        return h


class ShiftedSoftplus(Module):
    """Softplus with one log2 intercept."""

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


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


class AddEdge(object):
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

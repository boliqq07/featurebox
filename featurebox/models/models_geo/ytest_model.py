import torch

from torch import nn
from torch.nn import Linear, ModuleList

from torch_geometric.nn import CGConv
from torch_scatter import scatter


class Tes(torch.nn.Module):
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
                 simple_edge=True,
                 interactions=None,
                 readout_layer=None,
                 add_state=False,
                 de=True
                 ):

        super(Tes, self).__init__()

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
        self.simple_edge = simple_edge
        self.simple_z = simple_z
        self.interactions = interactions
        self.readout_layer = readout_layer
        self.out_size = out_size
        # 嵌入原子属性，备用
        # (嵌入别太多，容易慢，大多数情况下用不到。)
        # atomic_mass = torch.from_numpy(ase_data.atomic_masses)  # 嵌入原子质量
        # covalent_radii = torch.from_numpy(ase_data.covalent_radii)  # 嵌入共价半径
        # self.register_buffer('atomic_mass', atomic_mass)
        # self.register_buffer('atomic_radii', covalent_radii)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。

        # 定义输入
        # 使用原子性质,或者使用Embedding 产生随机数据。
        # 使用键性质,或者使用Embedding 产生随机数据。
        # if num_embeddings < 120:
        #     print("default, num_embeddings>=120,if you want simple the net work and "
        #           "This network does not apply to other elements, the num_embeddings could be less but large than "
        #           "the element type number in your data.")

        # 原子个数，一般不用动，这是所有原子种类数，
        # 一般来说，采用embedding的网络，
        # 在向其他元素（训练集中没有的）数据推广的能力较差。

        # self.embedding_e = Linear(2, 2)
        self.embedding_l = Linear(num_node_features, hidden_channels)
        self.embedding_l2 = Linear(hidden_channels, hidden_channels)
        self.distance_expansion = Linear(1, num_gaussians - num_bond_features)
        # 交互层 需要自定义
        # if interactions is None:

        self.lin0 = Linear(hidden_channels, num_filters)
        self.interactions = CGConv(channels=num_filters, dim=num_gaussians,
                        aggr='add', batch_norm=True,
                        bias=True, )



        # 合并层 需要自定义
        # if readout_layer is None:
        #     self.get_readout_layer()

        # self.register_buffer('initial_atomref', atomref)
        # self.atomref = None

            # 单个原子的性质是否加到最后

        self.add_state = add_state

        # self.reset_parameters()
        self.readout = readout
        self.lin1 = Linear(num_filters, num_filters )
        self.s1 = nn.Softplus()
        self.lin2 = Linear(num_filters , num_filters)
        self.s2 = nn.Softplus()
        self.lin3 = Linear(num_filters, out_size)

    def forward(self, data):
        # 使用embedding 作为假的原子特征输入，而没有用原子特征输入
        # assert hasattr(data, "z")
        # assert hasattr(data, "pos")
        # assert hasattr(data, "batch")

        batch = data.batch
        # 处理数据阶段
        x = data.x
        # h = F.relu(self.embedding_l(x))
        # h = self.embedding_l2(h)
        h=x

        # edge_weight = data.edge_weight

        # edge_attr = self.distance_expansion(edge_weight.view(-1, 1))
        # edge_attr = torch.cat((data.edge_attr, edge_attr), dim=1)
        # edge_index = data.edge_index
        # 自定义
        h=self.lin0(h)
        # h = self.interactions(h, edge_index, edge_attr,
        #                           )

        # h = self.lin1(h)
        # h = self.s1(h)
        device = torch.device("cpu")
        h= h.to(device = device)
        batch= batch.to(device = device)
        h = scatter(h, batch, dim=0, reduce=self.readout)
        h= h.to(device = torch.device("cuda:1"))
        h = self.lin2(h)
        h = self.s2(h)
        h = self.lin3(h)

        return h.view(-1)

    # def get_interactions_layer(self):


    # def get_readout_layer(self):
    #     """This part shloud re-defined. And must add the ``readout_layer`` attribute.
    #
    #     Examples::
    #
    #         >>> self.readout_layer = torch.nn.Sequential(...)
    #
    #     Examples::
    #
    #         >>> ...
    #         >>> self.readout_layer = YourNet()
    #     """
    #     self.readout_layer = ReadOutLayer(num_filters=self.num_filters, readout=self.readout, out_size=self.out_size)

    # def output_forward(self, out):
    #
    #     # if not self.dipole and self.mean is not None and self.std is not None:
    #     #     out = out * self.std + self.mean
    #     #
    #     # if self.dipole:
    #     #     out = torch.norm(out, dim=-1, keepdim=True)
    #     #
    #     # if self.scale is not None:
    #     #     out = self.scale * out
    #
    #     return out.view(-1)

    # def internal_forward(self, h, z, pos, batch, *args, **kwargs):
    #
    #     # if self.dipole:
    #     #     # 加入偶极矩
    #     #     # Get center of mass.
    #     #     mass = self.atomic_mass[z].view(-1, 1)
    #     #     c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
    #     #     h = h * (pos - c[batch])
    #     #
    #     # if not self.dipole and self.atomref is not None:
    #     #     h = h + self.atomref(z)
    #
    #     return h

    # def __repr__(self):
    #     return (f'{self.__class__.__name__}('
    #             f'hidden_channels={self.hidden_channels}, '
    #             f'num_filters={self.num_filters}, '
    #             f'num_interactions={self.num_interactions}, '
    #             f'num_gaussians={self.num_gaussians}, '
    #             f'cutoff={self.cutoff})')

    # def reset_parameters(self):
    #     self.embedding_e.reset_parameters()
    #     self.embedding_l.reset_parameters()
    #     self.embedding_l2.reset_parameters()
    #     if self.atomref is not None:
    #         self.atomref.weight.data.copy_(self.initial_atomref)


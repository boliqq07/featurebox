import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

try:

    from featurebox.layer.graph import _merge_cpp
    mod = _merge_cpp.mod
    me_idx_cpp = mod.merge_idx
    me_idx_mean_cpp = mod.merge_idx_mean
    me_idx_sum_cpp = mod.merge_idx_sum
    me_idx_max_cpp = mod.merge_idx_max
    me_idx_min_cpp = mod.merge_idx_min
except BaseException:
    mod = None

try:
    from torch_scatter import scatter_max
    from torch_scatter import scatter_add
    from torch_scatter import scatter_min
    from torch_scatter import scatter_mean

    scatter_mod = True
except ImportError:
    scatter_mod = None


class BaseLayer(nn.Module):
    """
    Base operation.
    """

    def __init__(self):
        super(BaseLayer, self).__init__()
        if mod is not None:
            self.merge_idx = self.merge_idx_cpp
        # elif scatter_mod is not None: # not very fast
        #     self.merge_idx = self.merge_idx_scatter
        else:
            self.merge_idx = self.merge_idx_py

    def merge_idx_methods(self, nbr_fea, node_atom_idx, methods=("mean", "max")):
        return torch.cat([self.merge_idx_py(nbr_fea, node_atom_idx, methodi) for methodi in methods], dim=-1)

    @staticmethod
    def merge_idx_py(nbr_fea, node_atom_idx, method="mean"):
        """
        Base realization.

        Parameters
        ----------
        nbr_fea:Tensor
        node_atom_idx:list
        method: callable,str

        Returns
        -------

        """
        # assert sum([len(idx_map) for idx_map in node_atom_idx]) == nbr_fea.data.shape[0]

        if method == "sum":
            temp = torch.cat([torch.sum(nbr_fea[i], dim=0, keepdim=True) for i in node_atom_idx])
        elif method == "max":
            temp = torch.cat([torch.max(nbr_fea[i], dim=0, keepdim=True)[1] for i in node_atom_idx])
        elif method == "min":
            temp = torch.cat([torch.min(nbr_fea[i], dim=0, keepdim=True)[1] for i in node_atom_idx])
        else:
            temp = torch.cat([torch.mean(nbr_fea[i], dim=0, keepdim=True) for i in node_atom_idx])

        return temp

    @staticmethod
    def merge_idx_cpp(nbr_fea, node_atom_idx, method="mean"):
        """Cpp realization,The first import is quite time consuming,
        but faster than *py realization in later call."""

        # assert sum([len(idx_map) for idx_map in node_atom_idx]) == nbr_fea.data.shape[0]
        return me_idx_cpp(nbr_fea, node_atom_idx, method)

    @staticmethod
    def merge_idx_cpp2(nbr_fea, node_atom_idx, method="mean"):
        """Cpp realization,the same with merge_idx_cpp,The first import is quite time consuming,
        but faster than *py realization in later call."""
        if method == "sum":
            temp = me_idx_sum_cpp(nbr_fea, node_atom_idx)
        elif method == "max":
            temp = me_idx_max_cpp(nbr_fea, node_atom_idx)
        elif method == "min":
            temp = me_idx_min_cpp(nbr_fea, node_atom_idx)
        else:
            temp = me_idx_mean_cpp(nbr_fea, node_atom_idx)

        return temp

    @staticmethod
    def merge_idx_scatter(nbr_fea, node_atom_idx, method="mean"):
        """torch-scatter realization,need torch-scatter model, it is useful in out-of-order index,
         but Not very fast in this part"""

        assert sum([len(idx_map) for idx_map in node_atom_idx]) == nbr_fea.data.shape[0]
        groups = torch.cat([torch.full((len(i),), n) for n, i in enumerate(node_atom_idx)])

        if method == "sum":
            temp = scatter_add(nbr_fea, groups, dim=0)
        elif method == "max":
            temp = scatter_max(nbr_fea, groups, dim=0)
        elif method == "min":
            temp = scatter_min(nbr_fea, groups, dim=0)
        else:
            temp = scatter_mean(nbr_fea, groups, dim=0)
        return temp

    @staticmethod
    def expand_idx(a, node_atom_idx):
        s2 = a.shape[1]
        lens = [len(i) for i in node_atom_idx]
        a = [ai.expand(i, s2) for i, ai in zip(lens, a)]
        return torch.cat(a, dim=0)


class MergeLayer(nn.Module):

    def forward(self, x_atom_fea, node_atom_idx):
        x_atom_fea_sum = self.pooling(x_atom_fea, node_atom_idx)
        return torch.mean(x_atom_fea_sum, dim=0, keepdim=True)

    def pooling(self, atom_fea, node_atom_idx):
        """
        Pooling the atom features to crystal features.\n
        N: Total number of atoms in the batch.\n
        N0: Total number of crystals in the batch.

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        node_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in node_atom_idx]
        return torch.cat(summed_fea, dim=0)


class UserLinear(nn.Module):
    __constants__ = ['max_in_features', 'out_features']

    max_in_features: int
    out_features: 1
    weight: Tensor

    def __init__(self,
                 max_in_features: int = 100, out_features: int = 20, bias: bool = True) -> None:
        super().__init__()
        self.max_in_features = max_in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, max_in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        last_shape = input.shape[-1]
        assert last_shape < self.max_in_features, "The last axis size of input should be smaller tan max_in_features" \
                                                  "but they are {}>{}".format(last_shape, self.max_in_features)
        return F.linear(input, self.weight[..., :last_shape], self.bias[:last_shape])

    def extra_repr(self) -> str:
        return 'max_in_features={}, out_features={}, bias={}'.format(
            self.max_in_features, self.out_features, self.bias is not None
        )


class MergeLayerAgg(MergeLayer):
    def __init__(self, inner_number=20):
        super().__init__()

        self.tup = nn.ModuleList(
            (nn.Linear(1, inner_number),
             nn.Linear(2, inner_number),
             nn.Linear(3, inner_number),
             nn.Linear(4, inner_number),
             nn.Linear(5, inner_number),
             nn.Linear(6, inner_number),
             nn.Linear(7, inner_number),
             nn.Linear(8, inner_number),
             nn.Linear(9, inner_number),
             nn.Linear(10, inner_number),
             UserLinear(100, inner_number))
        )

    def forward(self, x_atom_fea, node_atom_idx):
        x_atom_fea_sum = self.pooling(x_atom_fea, node_atom_idx)
        x_atom_fea_sum2 = self.coef_pooling(x_atom_fea, node_atom_idx)
        x_atom_fea_sum = x_atom_fea_sum2 + x_atom_fea_sum
        return x_atom_fea_sum

    def coef_pooling(self, x_atom_fea, node_atom_idx):
        # x_atom_fea (N_atom,L_feature)
        # return (N_bond,L_feature)
        summed_fea = []
        for idx_map in node_atom_idx:
            if len(idx_map) - 1 <= 9:
                x = x_atom_fea[idx_map].T
                f = self.tup[len(idx_map) - 1]
                summed_feai = f(x).T

            else:
                x = x_atom_fea[idx_map]
                f = self.tup[-1]
                summed_feai = f(x)
            summed_feai = torch.sum(summed_feai, dim=0, keepdim=True)
            summed_fea.append(summed_feai)
        return torch.cat(summed_fea, dim=0)


if __name__ == "__main__":
    li = UserLinear(100)
    # index = [torch.arange(0,13),torch.arange(13,37)]
    a = torch.rand((37, 16)).T
    b = torch.rand((33, 16)).T
    a = a.to(torch.device("cuda:0"))
    li.to(torch.device("cuda:0"))
    resa = li(a)
    li.eval()

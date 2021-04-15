import numpy as np
import torch
import torch.nn as nn
from mgetool.tool import tt

# from bgnet.layer.graph.baselayer import BaseLayer
from featurebox.layer.graph.baselayer import BaseLayer


class AtomLayer(BaseLayer):

    """
    Convolutional operation on graphs.

    There are 3 length for first dim(axis) of preprocessing.

    N_atom > N_ele> N_node.

    1. N_node: This is consistent with the number of target. N_node==N_target.
        In brief, The  first dim(axis) of preprocessing should be reduced to N_node to calculated error in the final network.
        (Not is this Layer, but the total network).

        N_node is the min preprocessing number. which can be expand in to N_atom, and N_ele.

        Examples:
            new_data = self.expand_idx(preprocessing, node_atom_idx)
            new_data = self.expand_idx(preprocessing, node_ele_idx)

    2. N_ele: This is the number of type of atomic number.

        This type is not used very much.

        N_node is the min preprocessing number. which can be expand in to and N_atom merged to N_node.

        Examples:
            new_data = self.expand_idx(preprocessing, ele_atom_idx)
            new_data = self.merge_idx(preprocessing, node_ele_idx)

    3. N_atom (N): This is the number of all atom in batch preprocessing.

        The preprocessing with N could be merged in to N_node by node_atom_idx.

        Examples:
            new_data = self.merge_idx(preprocessing, node_atom_idx)
            new_data = self.merge_idx(preprocessing, ele_atom_idx)

    common preprocessing:
        atom_fea:(torch.Tensor) torch.float32, shape (N, atom_fea_len)
        nbr_fea:(torch.Tensor) torch.float32, shape (N, M, atom_fea_len) M default is 5.
        state_fea: (torch.Tensor) torch.float32, shape (N_node, state_fea_len)
        atom_nbr_idx: (torch.Tensor) torch.int64, shape (N, M) M default is 5.

        oe: (torch.Tensor) torch.float32, shape (N, 17) in atom_fea
        sgt: (torch.Tensor) torch.float32, shape (N_node, 6) in state_fea

        node_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.
        node_ele_idx: (list of torch.Tensor) torch.int64, each one shape is different.
        ele_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.

    Theoretically "oe" it should belong to atom_fea,
    The reason for separation form atom_fea is better handling.
    """

    def __init__(self, atom_fea_len, nbr_fea_len, state_fea_len=0):
        """
        Notes:
            N: Total number of atoms in the batch or can be called N_atom.
            M: Max number of neighbors (fixed).
        Args:
            atom_fea_len: (int) default 1.
            nbr_fea_len: (int) default 1.
            state_fea_len: (int),default 2.
        """

        super().__init__()

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.state_fea_len = state_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len + self.state_fea_len,
                                 2 * self.atom_fea_len)
        self.fc_full2 = nn.Linear(2 * self.atom_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx=None):
        """
        Notes:
            N: Total number of atoms in the batch or can be called N_atom.
            M: Max number of neighbors (fixed).
            atom_fea_len: default 1.
            nbr_fea_len: default 1.
            state_fea_len: default 2.
        Args:
            atom_fea: (torch.Tensor) shape (N, atom_fea_len)
                Atom features.
            nbr_fea: (torch.Tensor) shape (N, M, nbr_fea_len)
                Bond features of each atom's M neighbors.
            atom_nbr_idx: (torch.LongTensor) shape (N, M)
                Indices of M neighbors of each atom.
            state_fea: (torch.Tensor) shape (N_node, state_fea_len)
                state_features.
            node_atom_idx: (list of torch.Tensor)
                index for preprocessing.
        Returns
            atom_out_fea: nn.Variable shape (N, atom_fea_len)
              refreshed atom features.
        """

        N, M = atom_nbr_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[atom_nbr_idx, :]

        if state_fea is not None and self.state_fea_len:
            state_atom_ed = self.expand_idx(state_fea, node_atom_idx)
            total_nbr_fea = torch.cat(
                [atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                 state_atom_ed.unsqueeze(1).expand(N, M, self.state_fea_len),
                 atom_nbr_fea, nbr_fea], dim=2)
        else:
            total_nbr_fea = torch.cat(
                [atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                 atom_nbr_fea, nbr_fea], dim=2)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.fc_full2(total_gated_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        prop_filter, prop_core = total_gated_fea.chunk(2, dim=2)
        prop_filter = self.sigmoid(prop_filter)
        prop_core = self.softplus1(prop_core)
        prop_core = torch.sum(prop_filter * prop_core, dim=1)
        prop_core = self.bn2(prop_core)
        out = self.softplus2(atom_fea + prop_core)
        return out


# if __name__ == "__main__":
#     torch.nn.LayerNorm
#     from bgnet.preprocessing.generator import GraphGenerator, MGEDataLoader
#     from bgnet.layer.graph.test.test_sumlayer import check_get
#
#     check_data = check_get("test/input_data206")
#     data_size = len(check_data[0])
#     gen = GraphGenerator(*check_data, targets=np.arange(data_size))
#     # gen = GraphGenerator(*check_data, targets=None)
#
#     loader = MGEDataLoader(
#         dataset=gen,
#         batch_size=13,
#         shuffle=False,
#         num_workers=0,
#     )
#
#     cl = AtomLayer(16, 1,2)
#
#     loader.to_cuda()
#     cl.to(torch.device("cuda:0"))
#
#     tt.t
#     for time, ((atom_fea,nbr_fea,state_fea,atom_nbr_idx,sgt,oe,node_atom_idx,node_ele_idx,ele_atom_idx),y) in enumerate(loader):
#         # if time ==30:
#         redata = cl(atom_fea, nbr_fea, atom_nbr_idx, state_fea, node_atom_idx)
#         # redata = cl(i[0])
#     tt.t
#     tt.p
import torch
import torch.nn as nn

# from bgnet.layer.graph.baselayer import BaseLayer
from featurebox.layer.graph.baselayer import BaseLayer


class StateLayer(BaseLayer):
    """
    Convolutional operation on graphs
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
                                 2 * self.state_fea_len)
        self.sigmoid = nn.Sigmoid()

        self.softplus1 = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx):
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

        # convolution
        atom_nbr_fea = atom_fea[atom_nbr_idx, :]

        atom_nbr_fea = torch.sum(atom_nbr_fea, dim=1, keepdim=False)

        node_nbr_fea_summed = self.merge_idx(atom_nbr_fea, node_atom_idx)  #

        node_fea = self.merge_idx(atom_fea, node_atom_idx)  #

        nbr_fea = torch.sum(nbr_fea, dim=1, keepdim=False)
        node_nbr_fea = self.merge_idx(nbr_fea, node_atom_idx)  #

        total_nbr_fea = torch.cat(
            [node_fea,
             node_nbr_fea_summed,
             node_nbr_fea, state_fea], dim=1)

        total_gated_fea = self.fc_full(total_nbr_fea)

        prop_filter, prop_core = total_gated_fea.chunk(2, dim=1)
        prop_filter = self.sigmoid(prop_filter)
        prop_core = prop_filter * prop_core

        out = self.softplus1(state_fea + prop_core)
        return out

# if __name__ == "__main__":
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
#     cl = StateLayer(16, 1,2)
#
#     # loader.to_cuda()
#     # cl.to(torch.device("cuda:0"))
#
#     tt.t
#     for time, ((atom_fea,nbr_fea,state_fea,atom_nbr_idx,sgt,oe,node_atom_idx,node_ele_idx,ele_atom_idx),y) in enumerate(loader):
#         # if time ==30:
#         redata = cl(atom_fea, nbr_fea, atom_nbr_idx, state_fea, node_atom_idx)
#         # redata = cl(i[0])
#     tt.t
#     tt.p

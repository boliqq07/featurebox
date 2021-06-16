"""This is one general script. For different data, you should re-write this and tune."""
import torch
from deprecated.classic import deprecated
from torch import nn

from featurebox.layer.graph.atomlayer import AtomLayer
from featurebox.layer.graph.baselayer import BaseLayer

@deprecated(version='0.1.0')
class Embed2(nn.Module):
    def __init__(self, nbr_fea_len=4, m1=None, m2=None):
        super(Embed2, self).__init__()
        # if m1 is None:
        #     m1 = 5 * nbr_fea_len
        # if m2 is None:
        #     m2 = int(m1 / 5)
        self.nbr_fea_len = nbr_fea_len
        self.linear = nn.Linear(nbr_fea_len, m1)
        self.relu = nn.ReLU()
        self.m2 = m2
        self.m1 = m1

    def forward(self, R: torch.Tensor):
        """

        Parameters
        ----------
        R:tensor,shape(N, n, nbr_fea_len)


        Returns
        -------

        """
        smooth = R[:, :, 0].unsqueeze(2)
        G = smooth.repeat(1, 1, self.nbr_fea_len)
        G = self.linear(G)
        G_bar = G[:, :, :self.m2]

        D = torch.bmm(G.transpose(1, 2), R)
        D = torch.bmm(D, R.transpose(1, 2))
        D = torch.bmm(D, G_bar)
        D = D.view((D.shape[0], -1))

        return D


class FeatureSymLayer(BaseLayer):
    def __init__(self, atom_fea_len, nbr_fea_len, in_state_fea_len, out_state_fea_len,
                 m1=None, m2=None):
        super().__init__()

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.state_fea_len = in_state_fea_len
        self.out_state_fea_len = out_state_fea_len

        # self.conv = nn.Conv2d(1, 2 * out_state_fea_len,kernel_size=3)

        self.fc_full = nn.Linear(m1 * m2,
                                 2 * out_state_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()

        self.embed = Embed2(nbr_fea_len=nbr_fea_len, m1=m1, m2=m2)

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx=None):
        """
        Notes:
            N: Total number of atoms in the batch or can be called N_atom.\n
            M: Max number of neighbors (fixed).\n
            atom_fea_len: default 1.\n
            nbr_fea_len: default 1.\n
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
        Returns:
            atom_out_fea: nn.Variable shape (N, atom_fea_len)
              refreshed atom features.
        """

        N, M = atom_nbr_idx.shape
        # convolution
        atom_nbr_fea = atom_fea[atom_nbr_idx, :]

        if state_fea is not None and self.state_fea_len:
            state_atom_ed = self.expand_idx(state_fea, node_atom_idx)
            total_nbr_fea = torch.cat(
                [nbr_fea,
                 atom_nbr_fea,
                 state_atom_ed.unsqueeze(1).expand(N, M, self.state_fea_len),
                 ], dim=2)
        else:
            total_nbr_fea = torch.cat(
                [nbr_fea, atom_nbr_fea], dim=2)

        new_state_fea = self.embed(total_nbr_fea)

        new_state_fea = self.merge_idx(new_state_fea, node_atom_idx)

        total_gated_fea = self.fc_full(new_state_fea)

        prop_filter, prop_core = total_gated_fea.chunk(2, dim=1)
        prop_filter = self.sigmoid(prop_filter)
        prop_core = prop_filter * prop_core

        out = self.softplus1(prop_core)
        return out


class Block2(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, state_fea_len, out_state_fea_len=None, m1=None, m2=None):
        if out_state_fea_len is None:
            out_state_fea_len = state_fea_len
        super().__init__()
        self.atomlayer = AtomLayer(atom_fea_len, nbr_fea_len, state_fea_len)
        self.statelayer = FeatureSymLayer(atom_fea_len, nbr_fea_len, state_fea_len, out_state_fea_len, m1, m2)

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx):
        atom_fea = self.atomlayer(atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx)
        state_fea = self.statelayer(atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx)
        return atom_fea, nbr_fea, state_fea


class SymNet(BaseLayer):
    def __init__(self, atom_fea_len, nbr_fea_len, state_fea_len,
                 inner_atom_fea_len=64, n_conv=2, h_fea_len=128, n_h=1,
                 classification=False, class_number=2):
        """

        Parameters
        ----------
        inner_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        inner_atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int,tuple
          Number of hidden features after merge to samples
        n_h: int
          Number of hidden layers after merge to samples
        """
        super().__init__()
        self.classification = classification
        self.mes = ("mean", "sum", "min", "max")
        self.mes = ("mean",)
        le = len(self.mes)
        if isinstance(h_fea_len, int):
            h_fea_len = tuple([h_fea_len for _ in range(n_h)])

        # change feature length
        self.embedding = nn.Linear(atom_fea_len, inner_atom_fea_len)
        # conv
        if isinstance(state_fea_len, int):
            state_fea_len = tuple([state_fea_len for _ in range(n_conv + 1)])
        else:
            assert len(state_fea_len) == n_conv + 1, "len(state_fea_len) == n_conv+1"

        cov = []
        for i in range(n_conv):
            cov.append(Block2(atom_fea_len=inner_atom_fea_len,
                              nbr_fea_len=nbr_fea_len,
                              state_fea_len=state_fea_len[i],
                              out_state_fea_len=state_fea_len[i + 1],
                              m1=10 * nbr_fea_len,
                              m2=nbr_fea_len,
                              ))

        self.convs = nn.ModuleList(cov)

        # self.merge_idx_methods expand self.mes times

        # conv to linear
        self.conv_to_fc = nn.Linear(le * inner_atom_fea_len, h_fea_len[0])
        self.conv_to_fc_softplus = nn.Softplus()
        n_h = len(h_fea_len)
        # linear (connect)
        if self.classification:
            self.dropout = nn.Dropout()
        if n_h >= 2:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len[hi], h_fea_len[hi + 1])
                                      for hi in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        # linear out
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, class_number)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_fea_len[-1], 1)

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx, *args, **kwargs):
        """
        Forward pass
        N: Total number of atoms in the batch.\n
        M: Max number of neighbors.\n
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        state_fea: (torch.Tensor) torch.float32, shape (N_node, state_fea_len)
            state feature.
        atom_nbr_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        node_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------
        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea, nbr_fea, state_fea = conv_func(atom_fea, nbr_fea,
                                                     state_fea=state_fea,
                                                     atom_nbr_idx=atom_nbr_idx,
                                                     node_atom_idx=node_atom_idx
                                                     )
        crys_fea = self.merge_idx_methods(atom_fea, node_atom_idx, methods=self.mes)
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)

        if self.classification:
            out = self.logsoftmax(out)
        return out

# class Embed(nn.Module):
#     def __init__(self, nbr_fea_len=4, m1=None, m2=None):
#         super(Embed, self).__init__()
#         if m1 is None:
#             m1 = 5 * nbr_fea_len
#         if m2 is None:
#             m2 = int(m1 / 5)
#         self.nbr_fea_len = 4
#         self.linear = nn.Linear(nbr_fea_len, m1)
#         self.relu = nn.ReLU()
#         self.m2 = m2
#         self.m1 = m1
#
#     def forward(self, R: torch.Tensor):
#         """
#
#         Parameters
#         ----------
#         R:tensor,shape(n, nbr_fea_len)
#
#
#         Returns
#         -------
#
#         """
#         smooth = R[:, 0].view((-1, 1))
#         G = smooth.repeat(1, self.nbr_fea_len)
#         # G = self.linear(G)
#         G_bar = G[:, :self.m2]
#         D = torch.mm(G.T, R)
#         D = torch.mm(D, R.T)
#         D = torch.mm(D, G_bar)
#         D = torch.flatten(D)
#         return D
#
#         # atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx = None

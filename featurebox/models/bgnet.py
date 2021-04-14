from __future__ import print_function, division

import torch
import torch.nn as nn

from bgnet.layer.graph.atomlayer import AtomLayer
from bgnet.layer.graph.baselayer import BaseLayer
from bgnet.layer.graph.bondlayer import BondLayer
from bgnet.layer.graph.disperselayer import InitAtomEnergy, DisperseDataLayer
from bgnet.layer.graph.interactlayer import InteractLayer


class Block(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, state_fea_len=0):
        super().__init__()
        self.atomlayer = AtomLayer(atom_fea_len, nbr_fea_len, state_fea_len)

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx):
        atom_fea = self.atomlayer(atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx)
        return atom_fea, nbr_fea, state_fea


class BgNet(BaseLayer):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, atom_fea_len, nbr_fea_len,
                 inner_atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False,class_number=2):
        """
        Initialize CrystalGraphConvNet.
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
        h_fea_len: int
          Number of hidden features after merge to samples
        n_h: int
          Number of hidden layers after merge to samples
        """
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(atom_fea_len, inner_atom_fea_len)
        self.convs = nn.ModuleList([Block(atom_fea_len=inner_atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])

        self.init_energy = InitAtomEnergy()

        self.interact = InteractLayer()

        self.disperse = DisperseDataLayer()

        self.conv_to_fc = nn.Linear(inner_atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        self.atomnbrstate = nn.Linear(h_fea_len,3)

        self.atomnbrstate2 = nn.Linear(21+h_fea_len, h_fea_len)
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, class_number)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout()

        self.relu_last = nn.ReLU()

    def forward(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx, sgt, oe, node_atom_idx, node_ele_idx, ele_atom_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        state_fea: (torch.Tensor) torch.float32, shape (N_node, state_fea_len)

        atom_nbr_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        oe: (torch.Tensor) torch.float32, shape (N, 17)

        sgt: (torch.Tensor) torch.float32, shape (N_node, 6) (N_node, 19)

        node_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.
            Mapping from the crystal idx to atom idx
        node_ele_idx: (list of torch.Tensor) torch.int64, each one shape is different.

        ele_atom_idx: (list of torch.Tensor) torch.int64, each one shape is different.



        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        int_energy = self.init_energy(atom_fea[:, 1].reshape(-1,1), node_atom_idx)

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea, nbr_fea, state_fea = conv_func(atom_fea, nbr_fea, state_fea, atom_nbr_idx, node_atom_idx)
        crys_fea = self.merge_idx_agg(atom_fea, node_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        crys_fea3 = self.atomnbrstate(crys_fea)

        os_interacted = self.interact(oe, node_ele_idx, ele_atom_idx)
        tune_energy = self.disperse(os_interacted, sgt, crys_fea3)
        # int_energy =
        crys_fea = torch.cat((crys_fea, tune_energy, int_energy), dim=1)

        crys_fea =self.atomnbrstate2(crys_fea)
        crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        out = self.relu_last(out)
        if self.classification:
            out = self.logsoftmax(out)
        return out
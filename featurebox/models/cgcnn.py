from __future__ import print_function, division

import torch.nn as nn

from bgnet.layer.graph.atomlayer import AtomLayer
from bgnet.layer.graph.baselayer import BaseLayer


class CrystalGraphConvNet(BaseLayer):
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
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(atom_fea_len, inner_atom_fea_len)
        self.convs = nn.ModuleList([AtomLayer(atom_fea_len=inner_atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(inner_atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
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

    def forward(self, atom_fea, nbr_fea, atom_nbr_idx, node_atom_idx):
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
            atom_fea = conv_func(atom_fea, nbr_fea, state_fea=None, atom_nbr_idx=atom_nbr_idx)
        crys_fea = self.merge_idx(atom_fea, node_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
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
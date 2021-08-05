"""This is one general script. For different data, you should re-write this and tune."""
from __future__ import print_function, division

import torch.nn as nn
from deprecated.sphinx import deprecated

from featurebox import AtomLayer
from featurebox import BaseLayer


@deprecated(version='0.1.0')
class CrystalGraphConvNet(BaseLayer):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.

    There are 3 length for first dim(axis) of data.

    `N_atom` > `N_ele`> `N_node`.

    1. `N_node`: This is consistent with the number of target. `N_node`==`N_target`. In brief, The first dim(axis) of data
       should be reduced to `N_node` to calculated error in the final network.

       `N_node` is the min data number. which can be expand in to `N_atom`, and `N_ele`.

       Examples:

            new_data = self.expand_idx(data, node_atom_idx)
            new_data = self.expand_idx(data, node_ele_idx)

    2. `N_ele`: This is the number of type of atomic number.

       This type is not used very much.

       `N_node` is the min data number. which can be expand in to and `N_atom` merged to `N_node`.

       Examples:

            new_data = self.expand_idx(data, ele_atom_idx)
            new_data = self.merge_idx(data, node_ele_idx)

    3. `N_atom` (`N`): This is the number of all atom in batch data.

       The data with `N` could be merged in to `N_node` by node_atom_idx.

       Examples:

            new_data = self.merge_idx(data, node_atom_idx)
            new_data = self.merge_idx(data, ele_atom_idx)

    Common data
    --------------

    `atom_fea`:(torch.Tensor) torch.float32, shape (N, atom_fea_len)

    `nbr_fea`:(torch.Tensor) torch.float32, shape (N, fill_size, atom_fea_len) M default is 5.

    `state_fea`: (torch.Tensor) torch.float32, shape (N_node, state_fea_len)

    `atom_nbr_idx`: (torch.Tensor) torch.int64, shape (N, fill_size) ,fill_size default is 5.

    `node_atom_idx`: (list of torch.Tensor) torch.int64, each one shape is different.

    `node_ele_idx`: (list of torch.Tensor) torch.int64, each one shape is different.

    `ele_atom_idx`: (list of torch.Tensor) torch.int64, each one shape is different.



    """

    def __init__(self, atom_fea_len, nbr_fea_len,
                 inner_atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, class_number=2):
        """

        Parameters
        ----------
        inner_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        inner_atom_fea_len: int >=1
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int,tuple
          Number of hidden features after merge to samples
        n_h: int
          Number of hidden layers after merge to samples
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        # self.mes = ("mean", "sum", "min", "max")
        self.mes = ("mean",)
        le = len(self.mes)
        if isinstance(h_fea_len, int):
            h_fea_len = tuple([h_fea_len for _ in range(n_h)])

        # change feature length
        self.embedding = nn.Linear(atom_fea_len, inner_atom_fea_len)
        # conv
        self.convs = nn.ModuleList([AtomLayer(atom_fea_len=inner_atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])

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
        state_fea:
            neglect.
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
        _ = state_fea
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, state_fea=None, atom_nbr_idx=atom_nbr_idx)
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


CGCNN = CrystalGraphConvNet  # alias

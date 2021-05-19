import itertools
from os import path


from sklearn.utils import check_random_state, shuffle
from featurebox.utils.fast._calculate_length import cal_length_numba
from featurebox.utils.fast._calculate_subp import subp_numba2d
from featurebox.utils.general import train_test

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    __doc__ = f.read()

from abc import abstractmethod
from itertools import zip_longest
from pathlib import Path
from typing import List, Sequence, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_convert

MODULE_DIR = Path(__file__).parent.absolute()


class _BaseGraphSingleGenerator(Dataset):

    def __init__(
            self,
            dataset_size: int,
            targets: np.ndarray,
            sample_weights: np.ndarray = None,
    ):
        """
        Args:
            dataset_size (int): Number of entries in dataset
            targets (ndarray): Feature to be predicted for each network
            sample_weights (npdarray): sample weights
        """
        if targets is not None:
            self.targets = np.array(targets).reshape((dataset_size, -1))
        else:
            self.targets = None

        if sample_weights is not None:
            self.sample_weights = np.array(sample_weights)
        else:
            self.sample_weights = None

        self.total_n = dataset_size
        self.mol_idx = np.arange(self.total_n)

    def __len__(self) -> int:
        return self.total_n

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    @staticmethod
    def _reform_data(
            *args: np.ndarray
    ) -> List:
        """
        1.add length of node_atom_idx and len of node_ele_idx,ele_atom_idx.
        2.transform the preprocessing to numpy with Fixed accuracy float32 and int64 .
        """

        inputs = list(args)
        inputs.append(args[0].shape[0])
        inputs.append(np.array(cal_length_numba(args[0][:, 0].astype(int))))
        inputs.append(inputs[-1].shape[0])
        # assert len(inputs[-2])>1

        return inputs

    def __getitem__(self, index: int) -> tuple:
        """please make sure always return a tuple!"""
        # Get the indices for this batch

        # Get the inputs for each batch
        inputs = self._generate_inputs(index)

        # Make the graph preprocessing
        inputs = self._reform_data(*inputs)

        # Return the batch
        if self.targets is None:
            return inputs,
        # get targets
        target_temp = self.targets[index]
        if self.sample_weights is None:
            return inputs, target_temp
        sample_weights_temp = self.sample_weights[index]

        return inputs, target_temp, sample_weights_temp

    def __setitem__(self, key, value):
        raise NotImplemented("This method is deleted in Dataset class.")

    @abstractmethod
    def _generate_inputs(self, batch_idx: int) -> Iterable:
        """Get the preprocessing by index"""


class GraphGenerator(_BaseGraphSingleGenerator):
    """
    A generator class that assembles several structures (indicated by
    batch_size) and form (x, y) pairs for model training.

    .. note:
        Add ["len_node_atom", "len_node_ele" "ele_atom_idx"]
        to distinguish the relationship between atoms and compounds.

    Returns:
        ["atom_fea","nbr_fea", "state_fea", "atom_nbr_idx",
        "len_node_atom", "len_node_ele" "ele_atom_idx"]
    """

    def __init__(
            self,
            atom_fea: List[np.ndarray],
            nbr_fea: List[np.ndarray],
            state_fea: List[np.ndarray],
            atom_nbr_idx: List[np.ndarray],
            *args,
            targets: np.ndarray = None,
            sample_weights: np.ndarray = None,
            **kwargs,
    ):
        """
        Base class has five probs, and not limit the number of attributes.

        More props would pass to arg, and named "other_prop_i_feature"
        or props would pass to kwarg, and named "other_prop_{...}_feature"

        Args:
            atom_fea: (list of np.array) list of atom feature matrix,
            nbr_fea: (list of np.array) list of bond features matrix
            state_fea: (list of np.array) list of [1, G] state features,
                where G is the global state feature dimension
                M is different for different structures
            atom_nbr_idx: (list of integer) list of (M, ) the other side atomic
                index of the bond, M is different for different structures,
                but it has to be the same as the corresponding index1.
            targets: (numpy array), N*1, where N is the number of structures
            sample_weights: (numpy array), N*1, where N is the number of structures
        """

        super().__init__(
            len(atom_fea), targets, sample_weights=sample_weights,
        )

        self.atom_fea = atom_fea
        self.nbr_fea = nbr_fea
        self.state_fea = state_fea

        self.atom_nbr_idx = atom_nbr_idx

        self.final_data_name = ["atom_fea", "nbr_fea", "state_fea", "atom_nbr_idx",
                                "node_atom_idx", "node_ele_idx", "ele_atom_idx"]
        self.add_prop = []
        for i, j in enumerate(args):
            name = "other_prop_{}_feature".format(i)
            self.__setattr__(name, j)
            self.final_data_name.append(name)
            self.add_prop.append(name)

        for i, j in kwargs.items():
            name = "other_prop_{}_feature".format(i)
            self.__setattr__(name, j)
            self.final_data_name.append(name)
            self.add_prop.append(name)

    def _generate_inputs(self, index: int):
        # Get the features and connectivity lists for this batch
        feature_list_temp = self.atom_fea[index]
        connection_list_temp = self.nbr_fea[index]
        global_list_temp = self.state_fea[index]
        atom_nbr_idx = self.atom_nbr_idx[index]

        return_list = [feature_list_temp, connection_list_temp, global_list_temp, atom_nbr_idx]

        for i in self.add_prop:
            try:
                return_list.append(self.__getattribute__(i)[index])
            except BaseException:
                raise TypeError("The {} is can not iterable".format(i))
        return return_list


class DuplicateGraphGenerator(_BaseGraphSingleGenerator):
    def __init__(
            self,
            atom_fea: List[np.ndarray],
            nbr_fea: List[np.ndarray],
            state_fea: List[np.ndarray],
            atom_nbr_idx: List[np.ndarray],
            targets: np.ndarray = None,
            sample_weights: np.ndarray = None,
            duplicate: int = 5,
            noise: float = 0,
            shuffle: bool = 0,
            random_state: bool = 0
    ):
        """
        The data are duplicated n times. Advise shuffle after processing.
        Base class has five probs, and not limit the number of attributes.

        More props would pass to arg, and named "other_prop_i_feature"
        or props would pass to kwarg, and named "other_prop_{...}_feature"

        Args:
            atom_fea: (list of np.array) list of atom feature matrix,
            nbr_fea: (list of np.array) list of bond features matrix
            state_fea: (list of np.array) list of [1, G] state features,
                where G is the global state feature dimension
                M is different for different structures
            atom_nbr_idx: (list of integer) list of (M, ) the other side atomic
                index of the bond, M is different for different structures,
                but it has to be the same as the corresponding index1.
            targets: (numpy array), N*1, where N is the number of structures
            sample_weights: (numpy array), N*1, where N is the number of structures
            duplicate: (int), times
            shuffle: (float), shuffle the atom rank in each compound in the duplicate data.
            random_state: (float), random seed
        """
        targets = np.repeat(np.array(targets), repeats=duplicate, axis=0)
        super().__init__(
            len(atom_fea) * duplicate, targets, sample_weights=sample_weights,
        )
        self.duplicate = duplicate
        self.noise = noise
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state)

        self.atom_fea, self.nbr_fea, self.state_fea, self.atom_nbr_idx = \
            self._repeat(atom_fea, nbr_fea, state_fea, atom_nbr_idx)

        self.final_data_name = ["atom_fea", "nbr_fea", "state_fea", "atom_nbr_idx",
                                "node_atom_idx", "node_ele_idx", "ele_atom_idx"]
        self.add_prop = []

    def _repeat(self, atom_fea, nbr_fea, state_fea, atom_nbr_idx):
        index = [np.arange(i.shape[0]) for i in atom_fea]
        if self.shuffle:
            indexes = [[shuffle(i, random_state=self.random_state) for i in index] for _ in range(self.duplicate - 1)]
            atom_fea_ = [a[i] for indexi in indexes for a, i in zip(atom_fea, indexi)]
            nbr_fea_ = [b[i] for indexi in indexes for b, i in zip(nbr_fea, indexi)]
            state_fea_ = [i for i in state_fea for _ in range(self.duplicate)]

            atom_nbr_idx_ = [subp_numba2d(a[i], i) for indexi in indexes for a, i in zip(atom_nbr_idx, indexi)]

            atom_fea_.extend(atom_fea)
            nbr_fea_.extend(nbr_fea)
            atom_nbr_idx_.extend(atom_nbr_idx)
            return atom_fea_, nbr_fea_, state_fea_, atom_nbr_idx_
        else:
            atom_fea_ = [i for i in atom_fea for _ in range(self.duplicate)]
            nbr_fea_ = [i for i in nbr_fea for _ in range(self.duplicate)]
            state_fea_ = [i for i in state_fea for _ in range(self.duplicate)]
            atom_nbr_idx_ = [i for i in atom_nbr_idx for _ in range(self.duplicate)]
            return atom_fea_, nbr_fea_, state_fea_, atom_nbr_idx_

    def _generate_inputs(self, index: int):
        # Get the features and connectivity lists for this batch
        feature_list_temp = self.atom_fea[index]
        connection_list_temp = self.nbr_fea[index]
        global_list_temp = self.state_fea[index]
        atom_nbr_idx = self.atom_nbr_idx[index]

        return_list = [feature_list_temp, connection_list_temp, global_list_temp, atom_nbr_idx]
        return return_list


def add_mark(data):
    """Change length to rank"""
    if isinstance(data, Tensor):
        data = data.numpy()
    if isinstance(data, list):
        data = np.array(data)

    a = np.cumsum(data)

    b = np.insert(a, 0, 0)[:-1]

    return [torch.arange(k, v) for k, v in zip(b, a)]


# def get_bond_node_idx(node_ele_idx, index1: Tensor):
#     spil = [index1[i] for i in node_ele_idx]
#     res = []
#     st = 0
#     for si in spil:
#         res.extend([torch.where(si == i)[0] + st for i in range(int(torch.max(si)) + 1)])
#         st += len(si)
#     res = [i.to(torch.int64) for i in res]
#     return res


class MGEDataLoader:
    """
    This loader:
        1.Add or transform the necessary indexes to declare the relationship between atoms and compounds.\n
        1.Transform data to torch.Tensor.\n
        2.Stack or cat the data belong the first axis(dim).\n

    If there are **N_node** samples.

    ===================  ==========================================   ============================
    Name                 N_node Graph data (Input)                    Batch graph data (Output)
    -------------------  ------------------------------------------   ----------------------------
    ``atom_fea``         (N, atom_fea_len)  * **N_node**              (N', atom_fea_len)
    ``nbr_fea``          (N, fill_size, atom_fea_len) * ***N_node**   (N', fill_size, atom_fea_len).
    ``state_fea``        (state_fea_len,)  * **N_node**               (N_node, state_fea_len,)
    ``atom_nbr_idx``     (N, fill_size) * **N_node**                  (N', fill_size)
    ===================  ==========================================   ============================

    Notes
    -----
        Though, the collate_fn parameter in torch.DataLoader could be used to collate preprocessing,
        but when we try to use it, we find it make the code difficult to read and maintain.
        thus, we just use "collate_fn=default_convert" to transform the preprocessing from numpy to tensor,
        and collate preprocessing outer of the torch.DataLoader

    Notes
    -----
        We replace the last 2 features in primary features to index list feature and add 1 features.\n
        len_node_atom -> node_atom_idx\n
        len_node_ele -> node_ele_idx\n
        ele_atom_idx -> node_ele_idx\n

    CrystalGraphDisordered,CrystalGraph,SingleMoleculeGraph in crystal, return 7 features.

    Return:
    ["atom_feature", "nbr_feature", "state_feature", 'atom_nbr_idx',
    "node_atom_idx", "node_ele_idx","ele_atom_idx"]

    other Graph self-defined.
    Return:
    ["atom_feature", "nbr_feature", "state_feature", 'atom_nbr_idx',
    ...
    "node_atom_idx", "node_ele_idx","ele_atom_idx"]

    """

    def __init__(self, dataset: GraphGenerator, collate_marks=None, batch_size=10, shuffle=True, num_workers=0,
                 **kwargs):
        """
        collate_marks:
                Transform all np.array to tensor, by stack or cat with self-defined,
                due to the elements of preprocessing in batch dont have consistent size.\n

                **Default is ('c', 'c', 's', 'c', 'f', 'c', 'f')**

                atom_fea "c",
                nbr_fea "c",
                state_fea "s",
                atom_nbr_idx = "c",

                ... other feature your defined.

                node_atom_idx = "f",
                node_ele_idx = "c",
                ele_atom_idx = "f",

        Args:
            dataset:(GraphGenerator)
            batch_size:(int)
            shuffle:(bool)
            num_workers:(int)
            kwargs:(Any) parameter for torch.DataLoader
            collate_marks:(tuple,list)

        """

        self.loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=default_convert, **kwargs)
        if collate_marks is None:
            collate_marks = ('c', 'c', 's', 'c', 'f', 'c', 'f')
        self.collate_marks = tuple(collate_marks)
        self._number_yield = 0
        self.device = torch.device('cpu')

    def __iter__(self):
        return self

    def __len__(self):
        return self.loader.__len__()

    def to_cuda(self, device='cuda:0'):
        """Change the data to cuda"""
        self.device = torch.device(device)

    def move(self, batch):
        """move data to cuda"""
        ba = []
        for i in batch:

            if isinstance(i, Tensor):
                ba.append(i.to(self.device))
            elif isinstance(i, Sequence):
                ba.append(self.move(i))
            else:
                raise TypeError("just accept list tuple of tensors or tensor")
        return ba

    def __next__(self):
        if self._number_yield < self.loader.__len__():
            batch = next(self.loader.__iter__())

            batch = self.collate_fn(batch, self.collate_marks)

            batch[0] = self.index_transform(batch[0])

            batch = self.move(batch)

            self._number_yield += 1
            return batch
        raise StopIteration

    def reset(self):
        self._number_yield = 0
        self.loader._iterator = self.loader._get_iterator()

    @staticmethod
    def collate_fn(batch: List, collate_marks: Tuple[str]) -> List[List]:
        """
        User collate function.

        Args:
            collate_marks: (tuple of str)
                the same size with features
            batch: (list)
                shape N_sample.
                Each one is a list or tuple,
                such as [[f1i,f2i],] or [[f1i,f2i], yi_value ] or [[f1i,f2i],yi_value, weight_i]

        Returns:
            such as [[f1,f2],] or [[f1,f2],y] or [[f1,f2],y,w]
        """

        def _deal_func(a, di="c"):
            """

            Args:
                a: (float,int,Tensor)
                    one feature with list format.
                di: str
                    collate type

            Returns:
                Tensor or list
            """
            elem = a[0]
            if isinstance(elem, float):
                a = torch.tensor(a, dtype=torch.float32)
                return torch.reshape(a, (-1, 1))
            elif isinstance(elem, int):
                a = torch.tensor(a, dtype=torch.int64)
                return torch.reshape(a, (-1, 1))
            elif isinstance(elem, Tensor):
                if elem.shape == ():
                    a = torch.stack(a, 0)
                    return torch.reshape(a, (-1, 1)) if len(a.shape) == 1 else a
                if di in ["c", "cat"]:

                    a = torch.cat(a, 0)

                    return torch.reshape(a, (-1, 1)) if len(a.shape) == 1 else a
                elif di in ["s", "stack"]:

                    a = torch.stack(a, 0)
                    return torch.reshape(a, (-1, 1)) if len(a.shape) == 1 else a
                else:
                    return a
            else:
                raise TypeError("just accept: Tensor, int, float")

        assert len(batch[0][0]) == len(
            collate_marks) or len(batch[0][0]) == len(
            collate_marks) + 3, "Must give one 'collate_marks' parameter and the shape is same with features.\n" \
                                "such as feature = [f1,f2,f3,f4,f5,f6,f7,f8,f9], without target" \
                                "The collate_marks = ('c', 'c', 's', 'c', 'f', 'c','f')\n" \
                                "'s' is torch.stack, 'c' is torch.cat, 'f' do noting but just return the list"

        transposed = list(zip(*batch))

        return [[_deal_func(si, di) for si, di in zip_longest(zip(*samples), collate_marks, fillvalue="f")]
                if i == 0 else _deal_func(samples, "c") for i, samples in enumerate(transposed)]

    @staticmethod
    def index_transform(data):
        """
        Transform the length of node_atom_idx and node_ele_idx to list of index.
        and add new node_bond_idx.

        [1,3,7,...] to  [[0,], [1,2,3], [4,5,6,7,8,9,10],...]

        Args:
            data: (List)
                batch preprocessing without collate.

        Returns:
            batch preprocessing with collate

        """

        data[-3] = add_mark(data[-3])  # node_atom_idx
        data[-2] = add_mark(data[-2])  # node_ele_idx
        data[-1] = add_mark(data[-1])  # ele_atom_idx

        data[0] = data[0].to(torch.float32)  # atom_fea
        data[1] = data[1].to(torch.float32)  # nbr_fea
        data[2] = data[2].to(torch.float32)  # state
        data[3] = data[3].to(torch.int64)  # atom_nbr_idx
        if len(data[1].shape) == 2:
            data[1] = data[1].unsqueeze(2)  # nbr_fea
        if len(data[0].shape) == 1:
            data[0] = data[0].unsqueeze(1)  # atom_fea

        return data


MEGDataLoader = MGEDataLoader
"""old name of MGEDataLoader"""


def get_train_test_loader(*data_X, data_y, train_size=None, test_size=0.25,
                          shuffle=False, random_state=0, stratify=None,
                          batch_size=50, dataLoader_shuffle=False,generator_type="GraphGenerator",
                          **kwargs):
    """Script to get train_loader and test_loader.You can customize and rewrite it.\n
    The parameters are from ``train_test`` and :class:`MGEDataLoader`
    """

    if generator_type=="GraphGenerator":
        generator_type=GraphGenerator
    else:
        generator_type=DuplicateGraphGenerator

    X_train_test, y_train_test, X_test, y_test = train_test(*data_X, data_y,
                                                            train_size=train_size,
                                                            test_size=test_size,
                                                            shuffle=shuffle,
                                                            random_state=random_state,
                                                            stratify=stratify)
    train_gen = generator_type(*X_train_test, targets=y_train_test)
    test_gen = generator_type(*X_test, targets=y_test)

    train_loader = MGEDataLoader(
        dataset=train_gen,
        batch_size=batch_size,
        shuffle=dataLoader_shuffle,
        **kwargs
    )
    test_loader = MGEDataLoader(
        dataset=test_gen,
        batch_size=batch_size,
        shuffle=dataLoader_shuffle,
        **kwargs
    )
    return train_loader, test_loader

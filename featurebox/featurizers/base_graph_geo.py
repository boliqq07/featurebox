"""
Abstract classes for building graph representations consist with ``pytorch-geometric``.

All the Graph in this part should return data as following:

Each Graph data (for each structure):


``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

``edge_attr``: Edge feature matrix. np.ndarray,  with shape [num_edges, num_edge_features]

``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

``state_attr``: state feature. np.ndarray, shape (1, num_state_features)

``z``: atom numbers. np.ndarray, with shape [num_nodes,]

Where the state_attr is added newly.

"""
import os
import warnings
from collections import Iterable
from shutil import rmtree
from typing import Dict, List

import numpy as np
import torch
import torch_geometric
from mgetool.tool import parallelize, batch_parallelize
from pymatgen.core import Structure

from featurebox.featurizers.base_transform import DummyConverter, BaseFeature, Converter, ConverterCat
from featurebox.featurizers.envir.environment import GEONNGet, env_method, env_names
from featurebox.featurizers.envir.local_env import NNDict
from featurebox.utils.look_json import get_marked_class


class _BaseStructureGraphGEO(BaseFeature):

    def __init__(self, collect=False, return_type="tensor", **kwargs):
        """

        Args:
            collect:(bool), Gather the batch data by different name.
            True Just for show!!
            return_type:(str), Return torch.tensor default, "numpy" is Just for show!!
            **kwargs:
        """
        super().__init__(**kwargs)
        self.graph_data_name = []
        if collect is False:
            self.get_collect_data = lambda x: x

        if collect is True:
            warnings.warn("The collect=True just used for temporary show!", UserWarning)
        if return_type != "tensor":
            warnings.warn("The collect != tensor just used for temporary display the shape of ndarray!!",
                          UserWarning)
        self.return_type = return_type
        self.collect = collect
        self.convert_funcs = [i for i in dir(self) if "_convert_" in i]

    def __add__(self, other):
        raise TypeError("There is no add.")

    def __call__(self, structure: Structure, **kwargs) -> Dict:
        return self.convert(structure, **kwargs)

    @staticmethod
    def _get_dummy_converter() -> DummyConverter:
        return DummyConverter()

    def _transform(self, structures: List[Structure], **kwargs):
        """

        Args:
            structures:(list) Preprocessing of samples need to transform to Graph.

            **kwargs:

        Returns:
            list of graphs:
                List of dict

        """
        assert isinstance(structures, Iterable)
        if hasattr(structures, "__len__"):
            assert len(structures) > 0, "Empty input data!"

        for i in kwargs.keys():
            if kwargs[i] is None:
                kwargs[i] = [None] * len(structures)

        iterables = zip(structures, *(kwargs.values()))

        if not self.batch_calculate:
            rets = parallelize(self.n_jobs, self._wrapper, iterables, tq=True)
            ret, self.support_ = zip(*rets)

        else:
            rets = batch_parallelize(self.n_jobs, self._wrapper, iterables, respective=False,
                                     tq=True, batch_size=self.batch_size)

            ret, self.support_ = zip(*rets)
        return ret

    def get_collect_data(self, graphs: List[Dict]):
        """
        Not used in default.

        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.
        Aim to get input for GraphGenerator.

        Args:
            graphs: list of dict
                list of graph dictionary for each structure
        """

        output = {}  # Will be a list of arrays

        # Convert the graphs to matrices
        for n, fi in enumerate(self.graph_data_name):
            np_data = [np.array(gi[fi]) if isinstance(gi, dict) else np.array(gi[n]) for gi in graphs]
            if self.return_type == "tensor":
                output[n] = [torch.from_numpy(i) for i in np_data]
            else:
                output[n] = np_data
        return output

    def transform(self, structures: List[Structure], **kwargs):
        """
        use ``convert`` to deal with batch of data.

        Args:
            structures: (list)
                preprocessing of samples need to transform to Graph.
            state_attributes: (list)
                preprocessing of samples need to add to Graph.
            y: (list)
                Target to train against (the same size with structure)
        Returns:
            data
        """

        return self.get_collect_data(self._transform(structures, **kwargs))

    def save(self, obj, name, root_dir="."):
        """Save."""
        torch.save(obj, os.path.join(root_dir, "raw", '{}.pt'.format(name)))

    def check_dup(self, structures, file_names="composition_name"):
        """Check the names duplication"""
        names = [i.composition.reduced_formula for i in structures]
        if file_names == "composition_name" and len(names) != len(set(names)):
            raise KeyError("There are same composition name for different structure, please use `rank_number`"
                           "to definition or specific names list.")
        elif len(set(file_names)) == len(structures):
            return file_names
        elif len(set(file_names)) == "rank_number":
            return ["raw_data_{}".format(i) for i in range(len(structures))]
        else:
            return names

    def transform_and_save(self, *args, root_dir=".", file_names="composition_name", save_mode="i"):
        r"""Save the data to 'root_dir/raw' """
        raw_path = os.path.join(root_dir, "raw")
        if os.path.isdir(raw_path):
            rmtree(raw_path)
        os.makedirs(raw_path)

        fns = self.check_dup(args[0], file_names=file_names)
        result = self.transform(*args)
        print("Save raw files to {}.".format(raw_path))
        if save_mode in ["I", "R", "i", "r", "Respective", "respective"]:
            [self.save(i, j, root_dir) for i, j in zip(result, fns)]
        else:
            self.save(result, "raw_data", root_dir=root_dir)
        print("Done.")
        return result

    def transform_and_to_data(self, *args) -> List[torch_geometric.data.Data]:
        """Return list of torch_geometric.data.Data."""
        from torch_geometric.data import Data
        result = self._transform(*args)
        return [Data.from_dict(i) for i in result]

    def convert(self, structure: Structure, **kwargs) -> Dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.

        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: list
                state attributes
            structure: Structure
                pymatgen Structure
            y:list
                Target

        Returns:
            dict
        """
        convert_funcs = [i for i in dir(self) if "_convert_" in i] if not self.convert_funcs else self.convert_funcs
        if len(convert_funcs) <= 1:
            raise NotImplementedError("Please implement the ``_convert_*`` functions, "
                                      "such as ``_convert_edge_attr``.")
        result_old = [getattr(self, i)(structure, **kwargs) for i in convert_funcs]
        result = {}
        [result.update(i) for i in result_old]
        if self.return_type == "tensor":
            result = {key: torch.from_numpy(value) for key, value in result.items() if key in self.graph_data_name}
        else:
            result = {key: value for key, value in result.items() if key in self.graph_data_name}
        return result

    def _convert_sample(self, *args, **kwargs):
        """
        Sample for convert.

        Returns:
            data:(dict)

        """

        return {}


class BaseStructureGraphGEO(_BaseStructureGraphGEO):
    """

    Returns

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

    ``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]

    Examples:
        >>> from torch_geometric.data.dataloader import DataLoader
        >>> sg1 = BaseStructureGraphGEO()
        >>> data_list = sg1.transform_and_to_data(structures_checked)
        >>> loader = DataLoader(data_list, batch_size=3)
        >>> for i in loader:
        ...     print(i)


    """

    def __init__(self,
                 atom_converter: Converter = None,
                 state_converter: Converter = None,
                 **kwargs):
        """

        Args:
            atom_converter: (BinaryMap) atom features converter. See Also:
                :class:`featurebox.featurizers.atom.mapper.AtomTableMap` , :class:`featurebox.featurizers.atom.mapper.AtomJsonMap` ,
                :class:`featurebox.featurizers.atom.mapper.AtomPymatgenPropMap`, :class:`featurebox.featurizers.atom.mapper.AtomTableMap`
            state_converter: (Converter) state features converter. See Also:
                :class:`featurebox.featurizers.state.state_mapper.StructurePymatgenPropMap`
                :mod:`featurebox.featurizers.state.statistics`
                :mod:`featurebox.featurizers.state.union`
            **kwargs:
        """
        super().__init__(**kwargs)

        self.atom_converter = atom_converter or self._get_dummy_converter()
        self.state_converter = state_converter or self._get_dummy_converter()
        self.graph_data_name = ["x", 'y', 'pos', "state_attr", 'z']

    def _convert_state_attr(self, structure, state_attributes=None, **kwargs):
        """return shape (1, num_state_features)"""
        _ = kwargs
        if state_attributes is not None:
            state_attributes = np.array(state_attributes)
        else:
            state_attributes = np.array([0.0, 0.0], dtype="float32")
        if isinstance(self.state_converter, DummyConverter):
            pass
        else:
            state_attributes = np.concatenate(
                (state_attributes, np.array(self.state_converter.convert(structure)).ravel()))

        state_attributes = state_attributes[np.newaxis, :]

        return {'state_attr': state_attributes}

    def _convert_y(self, structure, y=None, **kwargs):
        """return shape (1, num_state_features)"""
        _ = kwargs
        _ = structure
        if y is not None:
            y = np.array(y).ravel()[np.newaxis, :]
        else:
            y = np.array(1.0)

        return {'y': y}

    def _convert_x(self, structure, **kwargs):
        _ = kwargs
        z = np.array(structure.atomic_numbers).reshape(-1, 1)
        if isinstance(self.atom_converter, DummyConverter):
            atoms = self.atom_converter.convert(z)
        elif isinstance(self.atom_converter, ConverterCat):
            self.atom_converter.force_concatenate = True  # just accept the data could be concatenate as one array.
            atoms = self.atom_converter.convert(structure)
        else:
            atoms = self.atom_converter.convert(structure)
        return {'x': atoms, "z": z}

    def _convert_pos(self, structure, **kwargs):
        _ = kwargs
        return {'pos': structure.cart_coords}


class StructureGraphGEO(BaseStructureGraphGEO):
    """
    Returns

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]

    ``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

    ``edge_attr``: Edge feature matrix. np.ndarray,  with shape [num_edges, num_edge_features]

    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

    ``y``: target. np.ndarray, shape (1, num_target) , default shape (1,)

    ``state_attr``: state feature. np.ndarray, shape (1, num_state_features)

    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]

    Where the state_attr is added newly.

    Examples:
    >>> from torch_geometric.data.dataloader import DataLoader
    >>> sg1 = BaseStructureGraphGEO()
    >>> data_list = sg1.transform_and_to_data(structures_checked)
    >>> loader = DataLoader(data_list, batch_size=3)
    >>> for i in loader:
    ...     print(i)

    """

    def __init__(self, nn_strategy="find_points_in_spheres",
                 bond_generator=None,
                 bond_converter: Converter = None,
                 cutoff: float = 5.0,
                 **kwargs):
        """
        Args:
            nn_strategy: (str) NearNeighbor strategy.
                ["find_points_in_spheres", "find_xyz_in_spheres",
                "BrunnerNN_reciprocal", "BrunnerNN_real", "BrunnerNN_relative",
                "EconNN", "CrystalNN", "MinimumDistanceNNAll", "find_points_in_spheres","UserVoronoiNN"]
            atom_converter: (BinaryMap) atom features converter.
                See Also:
                :class:`featurebox.featurizers.atom.mapper.AtomTableMap` , :class:`featurebox.featurizers.atom.mapper.AtomJsonMap` ,
                :class:`featurebox.featurizers.atom.mapper.AtomPymatgenPropMap`, :class:`featurebox.featurizers.atom.mapper.AtomTableMap`
            bond_converter: (Converter)
                bond features converter, default=None.
            state_converter: (Converter)
                state features converter.
                See Also:
                :class:`featurebox.featurizers.state.state_mapper.StructurePymatgenPropMap`
                :mod:`featurebox.featurizers.state.statistics`
                :mod:`featurebox.featurizers.state.union`
            bond_generator: (GEONNGet, str)
                bond features converter.
            cutoff: (float)
                Whether to use depends on the ``nn_strategy``.
        """

        super().__init__(**kwargs)
        self.cutoff = cutoff

        if bond_generator is None:  # default use GEONNDict
            nn_strategy = get_marked_class(nn_strategy, NNDict)
            # there use the universal parameter, custom it please
            self.bond_generator = GEONNGet(nn_strategy,
                                           numerical_tol=1e-8, pbc=None, cutoff=cutoff)
        elif isinstance(bond_generator, str):  # new add "BaseDesGet"
            # todo
            self.nn_strategy = get_marked_class(nn_strategy, env_method[bond_generator])
            # there use the universal parameter, custom it please
            self.bond_generator = env_names[bond_generator](self.nn_strategy, self.cutoff,
                                                            numerical_tol=1e-8, pbc=None, cutoff=self.cutoff)
        else:  # defined BaseDesGet or BaseNNGet
            # todo
            self.bond_generator = bond_generator
            self.nn_strategy = self.bond_generator.nn_strategy

        self.bond_converter = bond_converter or self._get_dummy_converter()

        self.graph_data_name = ["x", 'edge_index', "edge_weight", "edge_attr", 'y', 'pos', "state_attr", 'z']

    def _convert_edges(self, structure, **kwargs):
        """get edge data."""
        _ = kwargs

        center_indices, edge_index, edge_attr, edge_weight, center_prop = self.bond_generator.convert(structure)

        if edge_weight.ndim == 2:
            if edge_weight.shape[1] == 3:
                edge_weight = np.sum(edge_weight ** 2, axis=1) ** 0.5
                edge_weight = edge_weight.reshape(-1, 1)
            elif edge_weight.shape[1] == 1:
                pass
            else:
                raise TypeError("edge_weight just accpet 'xyz' shape(n_node,3) or 'r' shape(n_node,)")
        elif edge_weight.ndim != 1:
            raise TypeError("edge_weight just accpet 'xyz' shape(n_node,3) or 'r' shape(n_node,1)")

        assert len(center_indices) == len(structure.atomic_numbers), \
            "center_indices less than atomic_numbers, this structure is illegal," \
            "which could due to there is discrete atom out of cutoff distance in structure."
        if not np.all(np.equal(np.sort(center_indices), np.array(center_indices))):
            raise UserWarning("center_indices rank is not compact with atomic_numbers,"
                              "which could due to the structure atoms rank are re-arranged in ``bond_generator``,"
                              "this could resulting in a correspondence error",
                              )

        edge_weight = self.bond_converter.convert(edge_weight)

        return {'edge_index': edge_index, "edge_weight": edge_weight, "edge_attr": edge_attr}
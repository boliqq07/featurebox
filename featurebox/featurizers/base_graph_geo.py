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

Where the state_attr is added newly.

"""
import os
import warnings
from collections import Iterable
from shutil import rmtree
from typing import Dict, List

import numpy as np
import torch
from mgetool.tool import parallelize, batch_parallelize
from pymatgen.core import Structure


from featurebox.featurizers.atom.mapper import BinaryMap
from featurebox.featurizers.base_transform import DummyConverter, BaseFeature, Converter, ConverterCat
from featurebox.featurizers.envir.environment import env_method, env_names, GEONNGet
from featurebox.featurizers.envir.local_env import NNDict
from featurebox.utils.look_json import get_marked_class


class StructureGraphGEO(BaseFeature):
    """
    This is a base class for converting converting structure into graphs or model inputs.

    Methods to be implemented are follows:
        convert(self, structure)
            This is to convert a structure into a graph dictionary.

    """

    def __init__(self, nn_strategy="find_points_in_spheres",
                 bond_generator=None,
                 atom_converter: Converter = None,
                 bond_converter: Converter = None,
                 state_converter: Converter = None,
                 return_bonds: str = "all",
                 cutoff: float = 5.0,
                 flatten: bool = False,
                 return_type="tensor",
                 **kwargs):
        """
        Parameters
        ----------
        nn_strategy : str
            NearNeighbor strategy
            ["find_points_in_spheres", "find_xyz_in_spheres",
            "BrunnerNN_reciprocal", "BrunnerNN_real", "BrunnerNN_relative",
            "EconNN", "CrystalNN", "MinimumDistanceNNAll", "find_points_in_spheres","UserVoronoiNN"]

        atom_converter: BinaryMap
            atom features converter.
            See Also:
            :class:`featurebox.featurizers.atom.mapper.AtomTableMap` , :class:`featurebox.featurizers.atom.mapper.AtomJsonMap` ,
            :class:`featurebox.featurizers.atom.mapper.AtomPymatgenPropMap`, :class:`featurebox.featurizers.atom.mapper.AtomTableMap`
        bond_converter : Converter
            bond features converter, default=None.
        state_converter : Converter
            state features converter.
            See Also:
            :class:`featurebox.featurizers.state.state_mapper.StructurePymatgenPropMap`
            :mod:`featurebox.featurizers.state.statistics`
            :mod:`featurebox.featurizers.state.union`
        bond_generator : GEONNGet, str
            bond features converter.

        return_bonds: "all","bonds","bond_state"
            which bond property return. default "all".
            ``"bonds_state"`` : bond properties and ``"bonds"`` : atoms number near this center atom.
        cutoff: float
            Whether to use depends on the ``nn_strategy``.
        **kwargs:

        """
        super().__init__(**kwargs)

        if flatten is False:
            self.get_flat_data = lambda x: x

        self.return_bonds = return_bonds
        self.cutoff = cutoff

        if bond_generator is None:  # default use GEONNDict
            nn_strategy = get_marked_class(nn_strategy, NNDict)
            # there use the universal parameter, custom it please
            self.bond_generator = GEONNGet(nn_strategy,
                                           numerical_tol=1e-8, pbc=None, cutoff=cutoff)
        elif isinstance(bond_generator, str):  # new add "BaseDesGet"
            self.nn_strategy = get_marked_class(nn_strategy, env_method[bond_generator])
            # there use the universal parameter, custom it please
            self.bond_generator = env_names[bond_generator](self.nn_strategy, self.cutoff,
                                                            numerical_tol=1e-8, pbc=None, cutoff=self.cutoff)
        else:  # defined BaseDesGet or BaseNNGet
            self.bond_generator = bond_generator
            self.nn_strategy = self.bond_generator.nn_strategy

        self.atom_converter = atom_converter or self._get_dummy_converter()
        self.bond_converter = bond_converter or self._get_dummy_converter()
        self.state_converter = state_converter or self._get_dummy_converter()
        self.graph_data_name = ["x", 'edge_index', "edge_attr", 'y', 'pos', "state_attr", ]
        self.cutoff = cutoff
        self.return_type = return_type
        if flatten is not False:
            warnings.warn("The flatten=True just used for temporary show!", UserWarning)
        if return_type != "tensor":
            warnings.warn("The return_type != tensor just used for temporary display the shape of ndarray!!",
                          UserWarning)

    def __add__(self, other):
        raise TypeError("There is no add.")

    def convert(self, structure: Structure, state_attributes: List = None, y=None) -> Dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.

        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Parameters
        ----------
        state_attributes: list
            state attributes
        structure: Structure
            pymatgen Structure
        y:list
            Target

        """
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

        if y is not None:
            y = np.array(y).ravel()[np.newaxis, :]
        else:
            y = np.array(1.0)

        center_indices, atom_nbr_idx, bond_states, bonds, center_prop = self.get_bond_fea(structure)
        if self.return_bonds == "all":
            bondss = np.concatenate((bonds, bond_states), axis=-1) if bonds is not None else bond_states
        elif self.return_bonds == "bonds":
            if bonds is not None:
                bondss = bonds
            else:
                raise TypeError
        elif self.return_bonds == "bonds_state":
            bondss = bond_states
        else:
            raise NotImplementedError()

        if isinstance(self.atom_converter, DummyConverter):
            atoms_numbers = np.array(structure.atomic_numbers)[center_indices].reshape(-1, 1)
            atoms = self.atom_converter.convert(atoms_numbers)
        elif isinstance(self.atom_converter, ConverterCat):
            self.atom_converter.force_concatenate = True  # just accept the data could be concatenate as one array.
            atoms = self.atom_converter.convert(structure)
            atoms = np.array([atoms[round(i)] for i in center_indices])
        else:
            atoms = self.atom_converter.convert(structure)
            atoms = np.array([atoms[round(i)] for i in center_indices])

        bonds = self.bond_converter.convert(bondss)

        # atoms number in the first column.
        if atoms.shape[1] == 1:
            if center_prop.shape[1] > 1:
                atoms = np.concatenate((np.array(atoms), center_prop), axis=1)
            else:
                pass
        elif atoms.shape[1] > 1:
            atoms_numbers = np.array(structure.atomic_numbers)[center_indices].reshape(-1, 1)
            if center_prop.shape[1] > 1:
                atoms = np.concatenate((atoms_numbers, np.array(atoms), center_prop), axis=1)
            else:
                atoms = np.concatenate((atoms_numbers, np.array(atoms)), axis=1)
        else:
            raise TypeError("Bad Converter for: atoms = self.atom_converter.convert(atoms_numbers)")

        pos = structure.cart_coords[center_indices]

        if self.return_type == "tensor":
            return {"x": torch.from_numpy(atoms),
                    "edge_index": torch.from_numpy(atom_nbr_idx),
                    "edge_attr": torch.from_numpy(bonds),
                    "y": torch.from_numpy(y),
                    "pos": torch.from_numpy(pos),
                    "state_attr": torch.from_numpy(state_attributes), }
        else:
            return {"x": atoms, "edge_index": atom_nbr_idx, "edge_attr": bonds, "y": y, "pos": pos,
                    "state_attr": state_attributes, }

    def get_bond_fea(self, structure: Structure):
        """
        Get atom features from structure, may be overwritten.
        """
        # assert hasattr(self.bond_generator, "convert")
        return self.bond_generator.convert(structure)

    def __call__(self, structure: Structure, *args, **kwargs) -> Dict:
        return self.convert(structure, *args, **kwargs)

    @staticmethod
    def _get_dummy_converter() -> DummyConverter:
        return DummyConverter()

    def _transform(self, structures: List[Structure], state_attributes: List = None, y=None, ):
        """

        Parameters
        ----------
        structures:list
            preprocessing of samples need to transform to Graph.
        state_attributes:list
            preprocessing of samples need to add to Graph.
        y:list
            Target to train against (the same size with structure)

        Returns
        -------
        list of graphs:
            List of dict

        """

        if state_attributes is None:
            state_attributes = [None] * len(structures)
        if y is None:
            y = [None] * len(structures)
        assert isinstance(structures, Iterable)
        if hasattr(structures, "__len__"):
            assert len(structures) > 0, "Empty input data!"
        iterables = zip(structures, state_attributes, y)

        if not self.batch_calculate:
            rets = parallelize(self.n_jobs, self._wrapper, iterables, tq=True)

            ret, self.support_ = zip(*rets)

        else:
            rets = batch_parallelize(self.n_jobs, self._wrapper, iterables, respective=False,
                                     tq=True, batch_size=self.batch_size)

            ret, self.support_ = zip(*rets)
        return ret

    def get_flat_data(self, graphs: List[Dict]) -> dict:
        """
        Not used in default.

        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.
        Aim to get input for GraphGenerator.

        Parameters
        ----------
        graphs: list of dict
            list of graph dictionary for each structure

        """
        output = {}  # Will be a list of arrays

        # Convert the graphs to matrices
        for n, fi in enumerate(self.graph_data_name):
            output[n] = [np.array(gi[fi]) if isinstance(gi, dict) else np.array(gi[n]) for gi in graphs]

        return output

    def transform(self, structures: List[Structure], state_attributes: List = None, y=None):
        """

        Parameters
        ----------
        structures:list
            preprocessing of samples need to transform to Graph.
        state_attributes:list
            preprocessing of samples need to add to Graph.
        y:list
            Target to train against (the same size with structure)

        """
        return self.get_flat_data(self._transform(structures, state_attributes, y=y))

    def save(self, obj, name, root_dir="."):
        """save."""
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

    def transform_and_to_data(self, *args):
        """Return list of torch_geometric.data.Data."""
        from torch_geometric.data import Data
        result = self._transform(*args)
        return [Data.from_dict(i) for i in result]

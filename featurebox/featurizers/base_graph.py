"""Abstract classes and utility operations for building graph representations and
preprocessing loaders.

All the Graph should return data as following:

Each Graph data (for each structure):

``atom_fea``: np.ndarray, shape (N, atom_fea_len)
    center properties.
``nbr_fea``: np.ndarray, shape (N, fill_size, atom_fea_len).
    neighbor_indexes for each center_index.
    `fill_size` default = 5.
``state_fea``: np.ndarray, shape (state_fea_len,)
    state feature.
``atom_nbr_idx``: np.ndarray, shape (N, fill_size)
    neighbor for each center, fill_size default is 5.

where N is number of atoms.
"""

import warnings
from operator import itemgetter
from typing import Union, Dict, List, Any

import numpy as np
from mgetool.tool import parallelize, batch_parallelize
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure

from featurebox.featurizers.atom.mapper import AtomPymatgenPropMap, AtomJsonMap, get_atom_fea_number, get_atom_fea_name
from featurebox.featurizers.base_transform import DummyConverter, BaseFeature, Converter
from featurebox.featurizers.bond.expander import BondGaussianConverter
from featurebox.featurizers.envir.environment import BaseNNGet, BaseDesGet, env_method, env_names
from featurebox.featurizers.envir.local_env import NNDict
from featurebox.utils.look_json import get_marked_class


def itemgetter_list(data_list: List, indices: List) -> tuple:
    """
    Get indices of data_list and return a tuple

    Args:
        data_list (list):  preprocessing list
        indices: (list) indices

    Returns:
        (tuple)
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return it(data_list),
    return it(data_list)


class _StructureGraph(BaseFeature):
    """
    Preferential use of _StructureGraphFixedRadius.
    This is a base class for converting converting structure into graphs or model inputs.

    Methods to be implemented are follows:
        convert(self, structure)
            This is to convert a structure into a graph dictionary
    """

    def __init__(self, nn_strategy: Union[str, NearNeighbors] = "find_points_in_spheres",
                 bond_generator: [BaseNNGet, BaseDesGet, str] = None,
                 atom_converter: Converter = None,
                 bond_converter: Converter = None,
                 state_converter: Converter = None,
                 return_bonds: str = "all",
                 cutoff: float = 5.0,
                 **kwargs):
        """
        Parameters
        ----------
        nn_strategy : str
            NearNeighbor strategy
            For bond_converter ="BaseNNGet": ["BrunnerNN_reciprocal", "BrunnerNN_real", "BrunnerNN_relative",
            "EconNN", "CrystalNN", "MinimumDistanceNNAll", "find_points_in_spheres","UserVoronoiNN"]
            For bond_converter ="BaseDesGet": ["ACSF","BehlerParrinello","EAD","EAMD","SOAP","SO3","SO4_Bispectrum","wACSF"]
            See Also:
            ``BaseNNGet`` :
            :class:`featurebox.featurizers.envir.local_env.MinimumDistanceNNAll`,
            ``BaseDesGet`` :
            :mod:`featurebox.featurizers.descriptors`,
            :class:`featurebox.featurizers.descriptors.SOAP.SOAP`,

        atom_converter: Converter
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
        bond_generator : _BaseEnvGet, str
            bond features converter.
            The function of this, is to convert data format to a fixed format.
            "BaseNNGet" or "BaseDesGet" or defined BaseNNGet,BaseDesGet object, default "BaseNNGet".
            1, BaseDesGet or 2, BaseDesGet. or there name.
            if object offered, rather str, the nn_strategy would use the nn_strategy in Converter.
            See Also:
            :class:`featurebox.featurizers.envir.environment.BaseNNGet` ,
            :class:`featurebox.featurizers.envir.environment.BaseDesGet`
        return_bonds: "all","bonds","bond_state"
            which bond property return. default "all".
            ``"bonds_state"`` : bond properties and ``"bonds"`` : atoms number near this center atom.
        cutoff: float
            Whether to use depends on the ``nn_strategy``.
        **kwargs:

        """

        super().__init__(**kwargs)
        self.return_bonds = return_bonds
        self.cutoff = cutoff

        if bond_generator is None:  # default use NNDict
            self.nn_strategy = get_marked_class(nn_strategy, NNDict)
            # there use the universal parameter, custom it please
            self.bond_generator = BaseNNGet(self.nn_strategy,
                                            numerical_tol=1e-8, pbc=None, cutoff=self.cutoff)
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
        self.graph_data_name = ["atom", "bond", "state", 'atom_nbr_idx']
        self.cutoff = cutoff

    def __add__(self, other):
        raise TypeError("There is no add.")

    def convert(self, structure: Structure, state_attributes: List = None) -> Dict:
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

        Returns
        -------
        ``atom_fea``: np.ndarray, shape (N, atom_fea_len)
            center properties.
        ``nbr_fea``: np.ndarray, shape (N, fill_size, atom_fea_len).
            neighbor_indexes for each center_index.
            `fill_size` default = 5.
        ``state_fea``: np.ndarray, shape (state_fea_len,)
             state feature.
        ``atom_nbr_idx``: np.ndarray, shape (N, fill_size)
            neighbor for each center, fill_size default is 5.
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
        center_indices, center_prop, atom_nbr_idx, bond_states, bonds = self.get_bond_fea(structure)
        if self.return_bonds == "all":
            bondss = np.concatenate((bond_states, bonds), axis=-1) if bonds is not None else bond_states
        elif self.return_bonds == "bonds":
            if bonds is not None:
                bondss = bonds
            else:
                raise TypeError
        elif self.return_bonds == "bonds_state":
            bondss = bond_states
        else:
            raise NotImplementedError()

        atoms = self.get_atom_fea(structure)
        atoms = [atoms[round(i)] for i in center_indices]
        atoms = self.atom_converter.convert(atoms)
        bonds = self.bond_converter.convert(bondss)

        # atoms number in the first column.
        if atoms.shape[1] == 1:
            if center_prop.shape[1] > 1:
                atoms = np.concatenate((np.array(atoms), center_prop), axis=1)
        else:
            atoms_numbers = np.array(structure.atomic_numbers)[center_indices].reshape(-1, 1)
            if center_prop.shape[1] > 1:
                atoms = np.concatenate((atoms_numbers, np.array(atoms), center_prop), axis=1)
            else:
                atoms = np.concatenate((atoms_numbers, np.array(atoms)), axis=1)

        return {"atom": atoms, "bond": bonds, "state": state_attributes, "atom_nbr_idx": atom_nbr_idx}

    @staticmethod
    def get_atom_fea(structure: Structure) -> List[Any]:
        """
        Get atom features from structure, may be overwritten for you self.
        """
        return get_atom_fea_number(structure)

    def get_bond_fea(self, structure: Structure):
        """
        Get atom features from structure, may be overwritten.
        """
        return self.bond_generator.convert(structure)

    def __call__(self, structure: Structure, *args, **kwargs) -> Dict:
        return self.convert(structure, *args, **kwargs)

    @staticmethod
    def _get_dummy_converter() -> DummyConverter:
        return DummyConverter()

    def _transform(self, structures: List[Structure], state_attributes: List = None):
        """

        Parameters
        ----------
        structures:list
            preprocessing of samples need to transform to Graph.
        state_attributes:List
            preprocessing of samples need to add to Graph.

        Returns
        -------
        list of graphs:
            List of dict

        """

        if state_attributes is None:
            state_attributes = [None] * len(structures)
        iterables = zip(structures, state_attributes)

        if not self.batch_calculate:
            rets = parallelize(self.n_jobs, self._wrapper, iterables, tq=True)

            ret, self.support_ = zip(*rets)

        else:
            rets = batch_parallelize(self.n_jobs, self._wrapper, iterables, respective=False,
                                     tq=True, batch_size=self.batch_size)

            ret, self.support_ = zip(*rets)
        return ret

    def get_flat_data(self, graphs: List[Dict]) -> tuple:
        """
        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.
        Aim to get input for GraphGenerator.

        Parameters
        ----------
        graphs: list of dict
            list of graph dictionary for each structure

        Returns
        -------
            tuple(node_features, edges_features, stats_values, atom_nbr_idx , ***)
        """
        output = []  # Will be a list of arrays

        # Convert the graphs to matrices
        for n, fi in enumerate(self.graph_data_name):
            output.append([np.array(gi[fi]) if isinstance(gi, dict) else np.array(gi[n]) for gi in graphs])

        return tuple(output)

    def transform(self, structures: List[Structure], state_attributes: List = None):
        """

        Parameters
        ----------
        structures:list
            preprocessing of samples need to transform to Graph.
        state_attributes
            preprocessing of samples need to add to Graph.

        Returns
        -------
        ``atom_fea``: list of np.ndarray, shape (N, atom_fea_len)
            center properties.
        ``nbr_fea``: list of np.ndarray, shape (N, fill_size, atom_fea_len).
            neighbor_indexes for each center_index.
            `fill_size` default = 5.
        ``state_fea``: list of np.ndarray, shape (state_fea_len,)
             state feature.
        ``atom_nbr_idx``: list of np.ndarray, shape (N, fill_size)
            neighbor for each center, fill_size default is 5.
        """
        return self.get_flat_data(self._transform(structures, state_attributes))


class _StructureGraphFixedRadius(_StructureGraph):
    """
    Preferential use of _StructureGraphFixedRadius.
    A phase radius is recommended: cutoff.
    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "find_points_in_spheres",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            state_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        if cutoff is None:
            warnings.warn(UserWarning("A phase radius is recommended: cutoff."))
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
            state_converter=state_converter, cutoff=cutoff, **kwargs)


class CrystalGraph(_StructureGraphFixedRadius):
    """
    CrystalGraph.

    Examples
    --------
    >>> cg1 = CrystalGraph()
    >>> d = cg1(structure)
    >>> d = cg1(structure, state_attributes=np.array([2,3.0]))
    >>> d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    >>> ds = cg1.fit_transform(structures)

    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "find_points_in_spheres",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            state_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
            state_converter=state_converter, cutoff=cutoff, **kwargs)


class CrystalGraphWithBondTypes(_StructureGraph):
    """
    Overwrite the bond attributes with bond types, defined simply by
    the metallicity of the atoms forming the bond. Three types of
    scenario is considered, nonmetal-nonmetal (type 0), metal-nonmetal (type 1), and
    metal-metal (type 2)

    Examples
    --------
    >>> cg1 = CrystalGraphWithBondTypes()
    >>> d = cg1(structure)
    >>> d = cg1(structure, state_attributes=np.array([2,3.0]))
    >>> d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    >>> ds = cg1.fit_transform(structures)

    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "VoronoiNN",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            state_converter: Converter = None,
            return_bonds="bonds",
            **kwargs
    ):
        if bond_converter is None:
            bond_converter = AtomPymatgenPropMap("is_metal", func=None, search_tp="number")

        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
                         state_converter=state_converter, return_bonds=return_bonds,
                         **kwargs)


class CrystalGraphDisordered(_StructureGraphFixedRadius):
    """
    Enable disordered site predictions

    Examples
    --------
    >>> cg1 = CrystalGraphDisordered()
    >>> d = cg1(structure)
    >>> d = cg1(structure, state_attributes=np.array([2,3.0]))
    >>> d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    >>> ds = cg1.fit_transform(structures)


    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "VoronoiNN",
            atom_converter: Converter = AtomJsonMap(),
            bond_converter: Converter = None,
            state_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff,
            state_converter=state_converter,
            **kwargs)

    @staticmethod
    def get_atom_fea(structure) -> List[dict]:
        """
        For a structure return the list of dictionary for the site occupancy
        for example, Fe0.5Ni0.5 site will be returned as {"Fe": 0.5, "Ni": 0.5}

        Args:
            structure (Structure): pymatgen Structure with potential site disorder

        Returns:
            a list of site fraction description
        """
        return get_atom_fea_name(structure)


class SimpleMolGraph(_StructureGraphFixedRadius):
    """
    Default using all atom pairs as bonds. The distance between atoms are used
    as bond features. By default the distance is expanded using a Gaussian
    expansion with centers at np.linspace(0, 4, 20) and width of 0.5

    Examples
    --------
    >>> cg1 = SimpleMolGraph()
    >>> d = cg1(structure)
    >>> d = cg1(structure, state_attributes=np.array([2,3.0]))
    >>> d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    >>> ds = cg1.fit_transform(structures)
    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "AllAtomPairs",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            state_converter: Converter = None,
            **kwargs):
        if bond_converter is None:
            bond_converter = BondGaussianConverter(np.linspace(0, 4, 20), 0.5)
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
                         state_converter=state_converter,
                         **kwargs)

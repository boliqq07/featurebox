
"""Abstract classes and utility operations for building graph representations and
preprocessing loaders."""

from operator import itemgetter
from typing import Union, Dict, List, Any

import numpy as np
from mgetool.tool import parallelize, batch_parallelize

from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure

from featurebox.featurizers.base_transform import DummyConverter, BaseFeature, Converter
from featurebox.featurizers.bond import BaseBondGet, BondGaussianConverter
from featurebox.featurizers.local_env import get_nn_strategy
from featurebox.featurizers.mapper import AtomPymatgenPropMap, AtomJsonMap


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

    def __init__(self, nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
                 atom_converter: Converter = None,
                 bond_converter: Converter = None,
                 **kwargs):
        """

        """
        super().__init__(**kwargs)

        self.nn_strategy = get_nn_strategy(nn_strategy)
        self.bond_generator = BaseBondGet(self.nn_strategy)
        self.atom_converter = atom_converter or self._get_dummy_converter()
        self.bond_converter = bond_converter or self._get_dummy_converter()
        self.graph_data_name = ["atom", "bond", "state", 'atom_nbr_idx']

    def convert(self, structure: Structure, state_attributes: List = None) -> Dict:
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.

        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        if state_attributes is not None:
            state_attributes = np.array(state_attributes)
        else:
            state_attributes = np.array([[0.0, 0.0]], dtype="float32")

        center_indices, atom_nbr_idx, _, bonds = self.get_bond_fea(structure)
        atoms = self.get_atom_fea(structure)
        atoms = self.atom_converter.convert(atoms)
        bonds = self.bond_converter.convert(bonds)
        return {"atom": atoms, "bond": bonds, "state": state_attributes, "atom_nbr_idx": atom_nbr_idx}

    @staticmethod
    def get_atom_fea(structure: Structure) -> List[Any]:
        """
        Get atom features from structure, may be overwritten

        Args:
            structure: (Pymatgen.Structure) pymatgen structure

        Returns:
            List of atomic numbers
        """
        return [i.specie.Z for i in structure]

    def get_bond_fea(self, structure: Structure):
        """
        Get atom features from structure, may be overwritten

        Args:
            structure: (Pymatgen.Structure) pymatgen structure
        """
        return self.bond_generator.convert(structure)

    def __call__(self, structure: Structure, *args, **kwargs) -> Dict:
        """
        Directly apply the converter to structure, alias to convert.

        Args:
            structure (Structure): input structure

        Returns (dict): graph dictionary

        """
        return self.convert(structure, *args, **kwargs)

    @staticmethod
    def _get_dummy_converter() -> "DummyConverter":
        return DummyConverter()

    def _transform(self, structures: List[Structure], state_attributes: List = None):
        """

        Args:
            structures:list
                preprocessing of samples need to transform to Graph.
            state_attributes
                preprocessing of samples need to add to Graph.

        Returns:
            list of graphs: List of dict

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

        Args:
            graphs: (list of dictionary) list of graph dictionary for each structure

        Returns:
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
        list of graphs: List of preprocessing
        """
        return self.get_flat_data(self._transform(structures, state_attributes))


class _StructureGraphFixedRadius(_StructureGraph):
    """
    Preferential use of _StructureGraphFixedRadius.
    This is a base class for converting converting structure into graphs or model inputs

    Methods to be implemented are follows:
        convert(self, structure)
            This is to convert a structure into a graph dictionary
    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        super(_StructureGraphFixedRadius, self).__init__(nn_strategy, atom_converter, bond_converter, **kwargs)
        self.cutoff = cutoff
        self.bond_generator = BaseBondGet(self.cutoff)


class CrystalGraph(_StructureGraphFixedRadius):
    """
    CrystalGraph.

    Examples
    ----------
    cg1 = CrystalGraph()
    d = cg1(structure)
    d = cg1(structure, state_attributes=np.array([2,3.0]))
    d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    or
    ds = cg1.fit_transform(structures)

    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        """
        Convert the structure into crystal graph.

        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=cutoff,
            **kwargs)


class CrystalGraphWithBondTypes(_StructureGraph):
    """
    Overwrite the bond attributes with bond types, defined simply by
    the metallicity of the atoms forming the bond. Three types of
    scenario is considered, nonmetal-nonmetal (type 0), metal-nonmetal (type 1), and
    metal-metal (type 2)

    Examples
    ----------
    cg1 = CrystalGraphWithBondTypes()
    d = cg1(structure)
    d = cg1(structure, state_attributes=np.array([2,3.0]))
    d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    # or
    ds = cg1.fit_transform(structures)

    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "VoronoiNN",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            **kwargs
    ):
        """

        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
        """
        if bond_converter is None:
            bond_converter = AtomPymatgenPropMap("is_metal", func=None, search_tp="number")

        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
                         **kwargs)


class CrystalGraphDisordered(_StructureGraphFixedRadius):
    """
    Enable disordered site predictions

    Examples
    ---------
    cg1 = CrystalGraphDisordered()
    d = cg1(structure)
    d = cg1(structure, state_attributes=np.array([2,3.0]))
    d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
     #or
    ds = cg1.fit_transform(structures)


    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
            atom_converter: Converter = AtomJsonMap(),
            bond_converter: Converter = None,
            cutoff: float = 7.0,
            **kwargs
    ):
        """
        Convert the structure into crystal graph

        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff,
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
        return [i.species.as_dict() for i in structure.sites]


class SimpleMolGraph(_StructureGraphFixedRadius):
    """
    Default using all atom pairs as bonds. The distance between atoms are used
    as bond features. By default the distance is expanded using a Gaussian
    expansion with centers at np.linspace(0, 4, 20) and width of 0.5

    Examples
    ----------
    cg1 = SimpleMolGraph()
    d = cg1(structure)
    d = cg1(structure, state_attributes=np.array([2,3.0]))
    d = cg1.convert(structure, state_attributes=np.array([2,3.0]))
    or
    ds = cg1.fit_transform(structures)

    """

    def __init__(
            self,
            nn_strategy: Union[str, NearNeighbors] = "AllAtomPairs",
            atom_converter: Converter = None,
            bond_converter: Converter = None,
            **kwargs):
        """
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atomic features converter object
            bond_converter (Converter): bond features converter object
        """
        if bond_converter is None:
            bond_converter = BondGaussianConverter(np.linspace(0, 4, 20), 0.5)
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter,
                         **kwargs)

"""Abstract classes for building graph representations consist with ``pytorch-geometric``.

All the Graph in this part should return data as following:

Each Graph data (for each structure):

``atom_fea``: np.ndarray, shape (N, atom_fea_len)
    atom properties.
``nbr_fea``: np.ndarray, shape (N_bond, bond_fea_len).
    properties for each edge pair.
``state_fea``: np.ndarray, shape (state_fea_len,)
    state feature.
``atom_nbr_idx``: np.ndarray, shape (2, N_bond)
    edge_index.

where N is number of atoms.
"""
from typing import Dict, List

from pymatgen.core import Structure

from featurebox.featurizers.base_graph import _StructureGraph
from featurebox.featurizers.base_transform import Converter
from featurebox.featurizers.envir.environment import GEONNGet
from featurebox.featurizers.envir.local_env import NNDict
from featurebox.utils.look_json import get_marked_class


class StructureGraphGEO(_StructureGraph):
    """
    This is a base class for converting converting structure into graphs or model inputs.

    Methods to be implemented are follows:
        convert(self, structure)
            This is to convert a structure into a graph dictionary


    Returns
    --------
    ``atom_fea``: np.ndarray, shape (N, atom_fea_len)
    atom properties.
    ``nbr_fea``: np.ndarray, shape (N_bond, bond_fea_len).
        properties for each edge pair.
    ``state_fea``: np.ndarray, shape (state_fea_len,)
        state feature.
    ``atom_nbr_idx``: np.ndarray, shape (2, N_bond)
        edge_index.

    """

    def __init__(self, nn_strategy="find_points_in_spheres",
                 bond_generator=None,
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
        if bond_generator is None:  # default use GEONNDict
            nn_strategy = get_marked_class(nn_strategy, NNDict)
            # there use the universal parameter, custom it please
            bond_generator = GEONNGet(nn_strategy,
                                      numerical_tol=1e-8, pbc=None, cutoff=cutoff)

        super().__init__(nn_strategy=nn_strategy,
                         bond_generator=bond_generator,
                         atom_converter=atom_converter,
                         bond_converter=bond_converter,
                         state_converter=state_converter,
                         return_bonds=return_bonds,
                         cutoff=cutoff,
                         **kwargs)

    def convert(self, structure: Structure, state_attributes: List = None) -> Dict:
        """

        Parameters
        ----------
        structure: pymatgen.core.Structure
        state_attributes:list
            state_attributes

        Returns
        --------
        ``atom_fea``: np.ndarray, shape (N, atom_fea_len)
        atom properties.
        ``nbr_fea``: np.ndarray, shape (N_bond, bond_fea_len).
            properties for each edge pair.
        ``state_fea``: np.ndarray, shape (state_fea_len,)
            state feature.
        ``atom_nbr_idx``: np.ndarray, shape (2, N_bond)
            edge_index.

        """
        return super().convert(structure, state_attributes)

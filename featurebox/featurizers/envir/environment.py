from pathlib import Path
from typing import Tuple, Union, Dict, List

import numpy as np
from ase import Atoms
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from featurebox.featurizers.base_transform import BaseFeature
from featurebox.featurizers.envir._get_radius_in_spheres import get_radius_in_spheres
from featurebox.featurizers.envir._get_xyz_in_spheres import get_xyz_in_spheres
from featurebox.featurizers.envir.desc_env import DesDict, universe_refine_des
from featurebox.featurizers.envir.local_env import NNDict, get_strategy_in_spheres, universe_refine_nn, geo_refine_nn
from featurebox.utils.look_json import get_marked_class
from featurebox.utils.predefined_typing import StructureOrMolecule, StructureOrMoleculeOrAtoms

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


class _BaseEnvGet(BaseFeature):
    """
    Get properties from Pymatgen.Structure.
    And each structure is convert to data as following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    ``center_properties``:np.ndarray of shape(n,l_c)
        center properties.
    """

    def __init__(self, n_jobs: int, on_errors: str = 'raise', return_type: str = 'any',
                 batch_calculate: bool = False,
                 batch_size: int = 30):
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type,
                         batch_calculate=batch_calculate,
                         batch_size=batch_size)
        self.adaptor_dict = {}  # Dynamic assignment
        self.adaptor = AseAtomsAdaptor()
        self.pbc = True
        self.cutoff = 5.0

    def ase_to_pymatgen(self, atom: Atoms, prop_dict=None):
        """ase_to_pymatgen"""
        if not prop_dict:
            prop_dict = self.adaptor_dict if self.adaptor_dict else {}

        structure = self.adaptor.get_structure(atom)
        if prop_dict != {}:
            [structure.__setattr__(k, v) for k, v in prop_dict.items()]
        return structure

    def pymatgen_to_ase(self, structure: Structure, prop_dict=None):
        """pymatgen_to_ase"""
        if not prop_dict:
            prop_dict = self.adaptor_dict if self.adaptor_dict else {}

        atoms = self.adaptor.get_atoms(structure)
        if prop_dict != {}:
            [atoms.__setattr__(k, v) for k, v in prop_dict.items()]
        return atoms

    @staticmethod
    def get_radius_in_spheres(
            structure: StructureOrMolecule, cutoff: float = 5.0,
            numerical_tol: float = 1e-8, pbc=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance
            pbc (bool):True

        Returns:
            center_indices, neighbor_indices, images, distances
        """
        return get_radius_in_spheres(structure, cutoff=cutoff, numerical_tol=numerical_tol, pbc=pbc)

    @staticmethod
    def get_xyz_in_spheres(
            structure: StructureOrMolecule, cutoff: float = 5.0,
            numerical_tol: float = 1e-8, pbc=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance
            pbc (bool):True
        Returns:
            center_indices, neighbor_indices, images, distances
        """
        return get_xyz_in_spheres(structure, cutoff=cutoff, numerical_tol=numerical_tol, pbc=pbc)


class BaseDesGet(_BaseEnvGet):
    """
    Get properties from Pymatgen.Structure.
    Where the nn_strategy is from ``pyXtal_FF``.
    And each structure is convert to data as following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    ``center_properties``:np.ndarray of shape(n,l_c)
        center properties.
    """

    def __init__(self, nn_strategy="SOAP", refine: str = None,
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None, cutoff=None, cut_off_name=None, check_align=True):
        """

        Parameters
        ----------
        nn_strategy:
            pyXtelff descriptors, which has calculate method.
            See Also:
            :mod:`featurebox.test_featurizers.descriptors`,
            :class:`featurebox.test_featurizers.descriptors.SOAP.SOAP`,
        refine:str
            sort method for neighbors of each atom.
            all the refine_method should return 5 result.
            See Also:
            :func:`universe_refine_des`
        refined_strategy_param:dict
            parameters for refine
        numerical_tol:float
            numerical_tol
        pbc:list
            periodicity in 3 direction
            3-length list,each one is 1 or 0. such as [0,0,0],The 1 mean in this direction is with periodicity.
        cutoff:float

        """

        super().__init__(n_jobs=1, on_errors='raise', return_type='any')
        if nn_strategy is None:
            nn_strategy = "SOAP"
        self.nn_strategy = get_marked_class(nn_strategy, DesDict)
        assert hasattr(self.nn_strategy, "calculate"), "The strategy must have calculate method"
        if refine is None:
            self.refine = universe_refine_des
        else:
            self.refine = after_treatment_func_map_des[refine]
        self.refined_strategy_param = refined_strategy_param if isinstance(refined_strategy_param, dict) else {}

        if cut_off_name:
            if cutoff:
                setattr(self.nn_strategy, cut_off_name, cutoff)
            self.cut_off_name = cut_off_name

        else:
            if cutoff is not None:
                if hasattr(nn_strategy, "cutoff"):
                    self.nn_strategy.cutoff = cutoff
                    self.cut_off_name = "cutoff"

                if hasattr(nn_strategy, "Rc"):
                    self.nn_strategy.Rc = cutoff
                    self.cut_off_name = "Rc"

                if hasattr(nn_strategy, "rcut"):
                    self.nn_strategy.rcut = cutoff
                    self.cut_off_name = "rcut"
                else:
                    raise TypeError("Can't find cut radius name")
            else:
                self.cut_off_name = cut_off_name

        self.cutoff = cutoff
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.check_align = check_align

    def convert(self, structure):
        """

        Args:
            structure:Structure,pymatgen Structure

        Returns:
            center_indices,center_prop,neighbor_indices,images,distances
        """

        return self.get_strategy(structure)

    def get_calculate_in_spheres(self, d):
        if isinstance(d, Structure):
            d = self.pymatgen_to_ase(d)

        d = self.nn_strategy.calculate(d)
        return d

    def get_strategy(self, structure: StructureOrMoleculeOrAtoms
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_stagy could be rewrite"""

        try:
            result_dict = self.get_calculate_in_spheres(structure)
            # todo add to refine
            # center_indices, neighbor_indices, images, distances = self._within_cutoff(structure, self.cutoff,
            #                                                                           self.numerical_tol)
            result = self.refine(result_dict)
        except ValueError as e:
            print(e)
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), getattr(self.nn_strategy, self.cut_off_name),
                getattr(self.nn_strategy, self.cut_off_name) + 2)
            )
            setattr(self.nn_strategy, self.cut_off_name, getattr(self.nn_strategy, self.cut_off_name) + 2)
            result_dict = self.get_calculate_in_spheres(structure)
            result = self.refine(result_dict)

        return result


class BaseNNGet(_BaseEnvGet):
    """
    Get properties from Pymatgen.Structure.
    Where the nn_strategy is from ``Pymatgen``.
    And each structure is convert to data as following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    ``center_indices``:np.ndarray of shape(n,l_c)
        center properties.
    """

    def __init__(self, nn_strategy: Union[NearNeighbors] = "UserVoronoiNN", refine: str = None,
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None, cutoff=5.0, check_align=True):
        """

        Parameters
        ----------
        nn_strategy: Union[NearNeighbors]
            search method for local_env for each atom.
            See Also:
            :class:`featurebox.test_featurizers.envir.local_env.MinimumDistanceNNAll`,
        refine:str
            sort method for neighbors of each atom.
            See Also:
            :func:`universe_refine_nn`
        refined_strategy_param:dict
            parameters for refine
        numerical_tol:float
            numerical_tol
        pbc:list
            only for find "find_points_in_spheres".
            periodicity in 3 direction
            3-length list,each one is 1 or 0. such as [0,0,0],The 1 mean in this direction is with periodicity.
        cutoff:
            if offered, the nn_strategy would be neglect and find neighbors using
                ``find_points_in_spheres`` in pymatgen.
        """

        super().__init__(n_jobs=1, on_errors='raise', return_type='any')
        self.nn_strategy = get_marked_class(nn_strategy, NNDict)
        if refine is None:
            self.refine = universe_refine_nn
        else:
            self.refine = after_treatment_func_map_nn[refine]
        self.refined_strategy_param = refined_strategy_param \
            if isinstance(refined_strategy_param, dict) else {}
        self.cutoff = cutoff
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.check_align = check_align

    def convert(self, structure: StructureOrMolecule) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            structure:Structure,pymatgen Structure

        Returns:
            center_indices,center_prop,neighbor_indices,images,distances
        """
        if self.nn_strategy == "find_points_in_spheres":
            return self.get_radius(structure)
        if self.nn_strategy == "find_xyz_in_spheres":
            return self.get_xyz(structure)
        else:
            return self.get_strategy(structure)

    def get_xyz(self, structure: StructureOrMolecule
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""

        cutoff = self.cutoff
        numerical_tol = self.numerical_tol

        result = self.get_xyz_in_spheres(structure, cutoff, numerical_tol)
        # Result: center_indices, neighbor_indices, images, distances, center_prop
        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param, )

        # Return: center_indices, neighbor_indices, images, distances, center_prop
        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return result
        else:
            self.cutoff += 2
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2))
            return self.get_xyz(structure)

    def get_radius(self, structure: StructureOrMolecule
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""

        cutoff = self.cutoff
        numerical_tol = self.numerical_tol

        result = self.get_radius_in_spheres(structure, cutoff, numerical_tol)
        # Result: center_indices, neighbor_indices, images, distances, center_prop
        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param, )

        # Return: center_indices, neighbor_indices, images, distances, center_prop
        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return result
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2))
            self.cutoff += 2
            return self.get_radius(structure)

    def get_strategy(self, structure: StructureOrMolecule
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_staagy could be rewrite."""
        # assert hasattr(self.nn_strategy, "cutoff")
        cutoff = self.cutoff
        numerical_tol = self.numerical_tol

        result = get_strategy_in_spheres(structure, self.nn_strategy, cutoff, numerical_tol)

        ele_numbers = np.array(structure.atomic_numbers)
        result = self.refine(*result, ele_numbers=ele_numbers, **self.refined_strategy_param)

        if len(result[0].tolist()) == len(structure.species) or not self.check_align:
            return result
        else:
            if hasattr(self.nn_strategy, "min_bond_distance"):
                print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                    str(structure.composition), self.nn_strategy.min_bond_distance,
                    self.nn_strategy.min_bond_distance + 0.1)
                )
                self.nn_strategy.min_bond_distance += 0.1
            else:
                print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                    str(structure.composition), self.nn_strategy.cutoff, self.nn_strategy.cutoff + 2)
                )
            self.nn_strategy.cutoff += 2
            self.cutoff += 2
            return self.get_strategy(structure)


class GEONNGet(BaseNNGet):

    def __init__(self, nn_strategy: Union[NearNeighbors] = "find_points_in_spheres",
                 refine="geo_refine_nn",
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None, cutoff=5.0, check_align=True):
        """

        Parameters
        ----------
        nn_strategy: Union[NearNeighbors]
            search method for local_env for each atom.
            See Also:
            :class:`featurebox.test_featurizers.envir.local_env.MinimumDistanceNNAll`,
        refine:str
            sort method for neighbors of each atom.
            See Also:
            :func:`universe_refine_nn`
        refined_strategy_param:dict
            parameters for refine
        numerical_tol:float
            numerical_tol
        pbc:list
            only for find "find_points_in_spheres".
            periodicity in 3 direction
            3-length list,each one is 1 or 0. such as [0,0,0],The 1 mean in this direction is with periodicity.
        cutoff:
            if offered, the nn_strategy would be neglect and find neighbors using
                ``find_points_in_spheres`` in pymatgen.
        """
        assert refine == "geo_refine_nn"
        super().__init__(nn_strategy=nn_strategy,
                         refine=refine,
                         refined_strategy_param=refined_strategy_param,
                         numerical_tol=numerical_tol, pbc=pbc,
                         cutoff=cutoff,
                         check_align=check_align,
                         )


#####################################################################################################################
after_treatment_func_map_des = {"universe_refine": universe_refine_des, "universe_refine_des": universe_refine_des}

after_treatment_func_map_nn = {"universe_refine": universe_refine_nn,
                               "universe_refine_nn": universe_refine_nn,
                               "geo_refine_nn": geo_refine_nn,
                               }
# class
env_names = {"BaseNNGet": BaseNNGet, "BaseDesGet": BaseDesGet, "GEONNGet": GEONNGet}
# local env method
env_method = {"BaseNNGet": NNDict, "BaseDesGet": DesDict, "GEONNGet": NNDict}
# after treatment
env_after_treatment_func_map = {"BaseNNGet": after_treatment_func_map_nn,
                                "BaseDesGet": after_treatment_func_map_des,
                                "GEONNGet": after_treatment_func_map_nn, }
#####################################################################################################################

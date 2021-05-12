from pathlib import Path
from typing import Tuple, Union, Dict, List

import numpy as np
from ase import Atoms
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.optimization.neighbors import find_points_in_spheres

from featurebox.featurizers.base_transform import BaseFeature
from featurebox.featurizers.envir.desc_env import DesDict
from featurebox.featurizers.envir.local_env import NNDict
from featurebox.utils.look_json import get_marked_class
from featurebox.utils.predefined_typing import StructureOrMolecule, StructureOrMoleculeOrAtoms

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


def universe_refine_des(d: Dict, fill_size=10, **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        d: dict, dict of descriptor. at lest contain "x" and "dxdr"
        fill_size: int

    Returns:
        (center_indices,center_prop, neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,1).\n
        center_prop: np.ndarray 1d(N,l_c).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size).

    """
    center, dxdr, seq = d["x"], d["dxdr"], d.get("seq")
    atom_len = center.shape[0]
    dxdr_len = dxdr.shape[0]

    if dxdr.ndim == 3:
        seq0 = seq[:, 0]

        uni = [len(seq[np.where(seq0 == i)]) for i in range(np.max(seq0) + 1)]
        if len(set(uni)) == 1:
            fill_size1 = int(dxdr_len / atom_len)
            dxdr_new = np.reshape(dxdr, (atom_len, fill_size1, -1))
            seq_len = seq.shape[0]
            fill_size2 = int(seq_len / atom_len)
            # assert fill_size2==fill_size1
            seq_new = np.reshape(seq[:, 1], (atom_len, fill_size2))
            left = fill_size - uni[0]
            if left == 0:
                pass
            elif left > 0:
                seq_new = np.concatenate((seq_new, np.array([seq_new[:, -1]] * left).T), axis=1)
                dxdr_new = np.concatenate((dxdr_new, np.array([dxdr_new[:, -1]] * left).transpose(1, 0, 2)), axis=1)

            else:
                seq_new = seq_new[:, :fill_size]
                dxdr_new = dxdr_new[:, :fill_size]

        else:
            seq_split = [list(seq[:, 1][np.where(seq0 == i)]) for i in range(np.max(seq0) + 1)]

            left = [fill_size - i for i in uni]
            seq_new = []
            dxdr_new = []
            for lefti, seqsi in zip(left, seq_split):
                if lefti == 0:
                    dxdri = dxdr[seqsi]
                elif lefti > 0:
                    seqsi.extend([seqsi[-1]] * lefti)
                    dxdri = dxdr[seqsi]
                else:
                    seqsi = seqsi[:fill_size]  # ? it is not loosely
                    dxdri = dxdr[:fill_size]

                seq_new.append(seqsi)
                dxdr_new.append(dxdri)
            seq_new = np.array(seq_new)
            dxdr_new = np.array(dxdr_new)

    elif dxdr.ndim == 4:
        dxdr_new = np.reshape(dxdr, (dxdr.shape[0], dxdr.shape[1], -1))
        seq_new = np.repeat(np.arange(atom_len).reshape(-1, 1), dxdr.shape[1], axis=1).T
        left = fill_size - atom_len
        if left == 0:
            pass
        elif left > 0:
            seq_new = np.concatenate((seq_new, np.array([seq_new[:, -1]] * left).T), axis=1)
            dxdr_new = np.concatenate((dxdr_new, np.array([dxdr_new[:, -1]] * left).transpose(1, 0, 2)), axis=1)

        else:
            seq_new = seq_new[:, :fill_size]
            dxdr_new = dxdr_new[:, :fill_size]
    else:
        raise NotImplementedError("Unrecognized data format")

    assert seq_new.shape[0] == center.shape[0]
    assert seq_new.shape[1] == dxdr_new.shape[1]
    return np.array(range(atom_len)), center, seq_new, dxdr_new, None


def universe_refine_nn(center_indices, neighbor_indices, distances, vectors=None, center_prop=None, ele_numbers=None,
                       fill_size=5,
                       dis_sort=False,
                       **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        center_indices: np.ndarray 1d
        neighbor_indices: np.ndarray 1d
        distances: np.ndarray 1d
        vectors: np.ndarray 2d
        fill_size: float
        dis_sort:bool
            sort neighbors with distance.

    Returns:
        (center_indices,center_indices,  neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,1).\n
        center_prop: np.ndarray 1d(N,l_c).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size,1).

    where l, and l_c >= 1
    """
    _ = ele_numbers

    neis = []
    diss = []
    vecs = []
    cen = np.array(sorted(set(center_indices.tolist())))
    for i in cen:
        # try:
        cidx = np.where(center_indices == i)[0]
        nei = neighbor_indices[cidx]
        disi = distances[cidx]
        if vectors is not None:
            vec = vectors[cidx, :]
        else:
            vec = None

        if dis_sort:
            neidx = np.argsort(disi)
            nei = nei[neidx]
            disi = disi[neidx]
            if vec is not None:
                vec = vec[neidx, :]

        if len(disi) >= fill_size:
            nei = nei[:fill_size]
            disi = disi[:fill_size]
            if vec is not None:
                vec = vec[:fill_size, :]
        else:
            while len(disi) < fill_size:
                nei = np.append(nei, nei[-1])
                disi = np.append(disi, disi[-1])
                if vec is not None:
                    vec = np.concatenate((vec, vec[-1, :].reshape(1, -1)), axis=0)

        neis.append(nei)
        diss.append(disi)

        if vectors is not None:
            vecs.append(vec)
    if vectors is not None:
        vecs = np.array(vecs)
    else:
        vecs = []
    if center_prop is None:
        return np.array(cen).ravel(), np.array(cen).reshape(-1, 1), np.array(neis), vecs, np.array(diss)[
            ..., np.newaxis]
    else:
        return np.array(cen).ravel(), center_prop, np.array(neis), vecs, np.array(diss)[..., np.newaxis]


def perovskite_refine_nn(center_indices, neighbor_indices, distances, vectors=None, center_prop=None,
                         ele_numbers=None, fill_size=5, dis_sort=False, **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        center_indices: np.ndarray 1d
        neighbor_indices: np.ndarray 1d
        distances: np.ndarray 1d
        vectors: np.ndarray 2d
        fill_size: float
        dis_sort:bool
            sort neighbors with distance.

    Returns:
        (center_indices,center_indices,  neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,1).\n
        center_prop: np.ndarray 1d(N,l_c).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size,1).

    where l, and l_c >= 1
    """
    # assert center_prop == None
    #
    # # todo
    #
    # return np.array(cen).ravel(), np.array(cen).reshape(-1, 1), np.array(neis), vecs, np.array(diss)[
    #         ..., np.newaxis]
    return


class _BaseEnvGet(BaseFeature):
    """
    Base class return features of each atoms, meanwhile it re-back bond feature.
    """

    def __init__(self, n_jobs: int, on_errors: str = 'raise', return_type: str = 'any',
                 batch_calculate: bool = False,
                 batch_size: int = 30):
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type,
                         batch_calculate=batch_calculate,
                         batch_size=batch_size)
        self.adaptor_dict = {}  # Dynamic assignment
        self.adaptor = AseAtomsAdaptor()

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


class BaseDesGet(_BaseEnvGet):
    """
    Get atoms features from Pymatgen.structure.
    Though, the class return features of each atoms, meanwhile it re-back bond feature.

    And get following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``center_indices``:np.ndarray of shape(n,l_c)
        center properties.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.
    """

    def __init__(self, nn_strategy="SOAP", refine: str = None,
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None, cutoff=None, cut_off_name=None):
        """

        Parameters
        ----------
        nn_strategy:
            pyXtelff descriptors, which has calculate method.
        refine:str
            sort method for neighbors of each atom. The key in ``refine_method``
            all the refine_method should return 5 result.
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

    def convert(self, structure):
        """

        Args:
            structure:Structure,pymatgen Structure

        Returns:
            center_indices,center_prop,neighbor_indices,images,distances
        """

        return self.get_graphs_with_strategy_merge_sort(structure)

    def _calculate(self, d):
        if isinstance(d, Structure):
            d = self.pymatgen_to_ase(d)

        d = self.nn_strategy.calculate(d)
        return d

    def get_graphs_with_strategy_merge_sort(self, structure: StructureOrMoleculeOrAtoms
                                            ) -> Tuple[np.ndarray]:
        """For get bond distance with different strategy, for different nn_stagy could be rewrite"""

        try:
            result_dict = self._calculate(structure)
            result = self.refine(result_dict)
        except ValueError as e:
            print(e)
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), getattr(self.nn_strategy, self.cut_off_name),
                getattr(self.nn_strategy, self.cut_off_name) + 2)
            )
            setattr(self.nn_strategy, self.cut_off_name, getattr(self.nn_strategy, self.cut_off_name) + 2)
            result_dict = self._calculate(structure)
            result = self.refine(result_dict)

        return result


class BaseNNGet(BaseFeature):
    """
    Get properties from Pymatgen.structure.
    And get following :

    ``center_indices``:np.ndarray of shape(n,)
        center indexes.
    ``center_indices``:np.ndarray of shape(n,l_c)
        center properties.
    ``neighbor_indices``:np.ndarray of shape(n,fill_size)
        neighbor_indexes for each center_index.
        `fill_size` is the parameter of `refine` function.
    ``images``:np.ndarray of shape(n,lb>=3)
        offset vector in 3 orientations or more bond properties.
    ``distances``:np.ndarray of shape(n,fill_size)
        distance of neighbor_indexes for each center_index.

    """

    def __init__(self, nn_strategy: Union[NearNeighbors] = "UserVoronoiNN", refine: str = None,
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None, cutoff=5.0):
        """

        Parameters
        ----------
        nn_strategy: Union[NearNeighbors]
            search method for local_env for each atom.
        refine:str
            sort method for neighbors of each atom. The key in ``bond.refine_method``
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
        self.refined_strategy_param = refined_strategy_param if isinstance(refined_strategy_param, dict) else {}
        self.cutoff = cutoff
        self.numerical_tol = numerical_tol
        self.pbc = pbc

    def convert(self, structure: StructureOrMolecule) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            structure:Structure,pymatgen Structure

        Returns:
            center_indices,center_prop,neighbor_indices,images,distances
        """
        if self.nn_strategy == "find_points_in_spheres":
            return self.get_graphs_within_cutoff_merge_sort(structure)
        else:
            return self.get_graphs_with_strategy_merge_sort(structure)

    def _within_cutoff(self,
                       structure: StructureOrMolecule, cutoff: float = 5.0, numerical_tol: float = 1e-8
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance

        Returns:
            center_indices, neighbor_indices, images, distances
        """
        if isinstance(structure, Structure):
            lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
            pbc = np.array([1, 1, 1], dtype=int)
        elif isinstance(structure, Molecule):
            lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
            pbc = np.array([0, 0, 0], dtype=int)
        else:
            raise ValueError("structure type not supported")
        if self.pbc is not None:
            pbc = np.array(self.pbc, dtype=int)
        r = float(cutoff)
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
        center_indices, neighbor_indices, images, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
        )
        center_indices = center_indices.astype(np.int)
        neighbor_indices = neighbor_indices.astype(np.int)
        images = images.astype(np.int)
        distances = distances.astype(np.float32)
        exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
        return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[
            exclude_self]

    def get_graphs_within_cutoff_merge_sort(self, structure: StructureOrMolecule
                                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""
        cutoff = self.cutoff
        numerical_tol = self.numerical_tol
        center_indices, neighbor_indices, images, distances = self._within_cutoff(structure, cutoff,
                                                                                  numerical_tol)
        ele_numbers = np.array(structure.atomic_numbers)
        center_indices, center_prop, neighbor_indices, images, distances = self.refine(center_indices, neighbor_indices,
                                                                                       distances,
                                                                                       images,
                                                                                       center_prop=None,
                                                                                       ele_numbers=ele_numbers,
                                                                                       **self.refined_strategy_param,
                                                                                       )

        if len(center_indices.tolist()) == len(structure.species):
            return center_indices, center_prop, neighbor_indices, images, distances
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2)
            )
            center_indices, center_prop, neighbor_indices, images, distances = self.get_graphs_within_cutoff_merge_sort(
                structure)
            return center_indices, center_prop, neighbor_indices, images, distances

    def get_graphs_with_strategy_merge_sort(self, structure: StructureOrMolecule
                                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_staagy could be rewrite."""
        # assert hasattr(self.nn_strategy, "cutoff")

        index1 = []
        index2 = []
        bonds = []
        center_prop = []

        image = []
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            index2i = [neighbor["site_index"] for neighbor in neighbors]
            bondsi = [neighbor["weight"] for neighbor in neighbors]
            imagei = [list(neighbor["image"]) + list(neighbor.get("add_image", (0,))) for neighbor in neighbors]
            center_propi = [neighbor.get("add_atom_prop", None) for neighbor in neighbors]
            index2.extend(index2i)
            image.extend(imagei)
            bonds.extend(bondsi)
            center_prop.append(center_propi)

        if None in center_prop[0] or [] in center_prop:
            center_prop = None
        else:
            center_prop = np.array(center_prop)

        ele_numbers = np.array(structure.atomic_numbers)
        center_indices, cen_prop, neighbor_indices, images, distances = self.refine(np.array(index1), np.array(index2),
                                                                                    np.array(bonds), np.array(image),
                                                                                    center_prop=center_prop,
                                                                                    ele_numbers=ele_numbers,
                                                                                    **self.refined_strategy_param)

        if len(center_indices.tolist()) == len(structure.species):
            return center_indices, cen_prop, neighbor_indices, images, distances
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
            center_indices, center_prop, neighbor_indices, images, distances = self.get_graphs_with_strategy_merge_sort(
                structure)

            return center_indices, cen_prop, neighbor_indices, images, distances


#####################################################################################################################
after_treatment_func_map_des = {"universe_refine": universe_refine_des}

after_treatment_func_map_nn = {"universe_refine": universe_refine_nn, "perovskite_refine_nn": perovskite_refine_nn}
# class
env_names = {"BaseNNGet": BaseNNGet, "BaseDesGet": BaseDesGet}
# local env method
env_method = {"BaseNNGet": NNDict, "BaseDesGet": DesDict, }
# after treatment
env_after_treatment_func_map = {"BaseNNGet": after_treatment_func_map_nn, "BaseDesGet": after_treatment_func_map_des, }
#####################################################################################################################

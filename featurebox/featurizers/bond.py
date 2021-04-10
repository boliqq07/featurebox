import numpy as np

from pathlib import Path
from typing import Tuple, Union, Dict, Callable, List

from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from featurebox.featurizers.base_transform import BaseFeature

from featurebox.utils.typing import StructureOrMolecule

MODULE_DIR = Path(__file__).parent.parent.absolute()


def universe_refine(center_indices, neighbor_indices, distances, vectors=None, fill_size=5, dis_sort=False, **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        center_indices: np.ndarray 1d
        neighbor_indices: np.ndarray 1d
        distances: np.ndarray 1d
        vectors: np.ndarray 1d
        fill_size: float
        dis_sort:bool
            sort neighbors with distance.

    Returns:
        (center_indices, neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size).

    """
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
                    vec = np.concatenate((vec, vec[-1, :].reshape(1, 3)), axis=0)

        neis.append(nei)
        diss.append(disi)

        if vectors is not None:
            vecs.append(vec)
    if vectors is not None:
        vecs = np.array(vecs)
    else:
        vecs = []

    return np.array(cen), np.array(neis), vecs, np.array(diss)


refine_method = {"universe_refine": universe_refine}


class BaseBondGet(BaseFeature):
    """
    Get bond from Pymatgen.structure.

    """

    def __init__(self, nn_strategy: Union[NearNeighbors, float], refine: str = None,
                 refined_strategy_param: Dict = None,
                 numerical_tol=1e-8, pbc: List[int] = None):
        """

        Parameters
        ----------
        nn_strategy: Union[NearNeighbors, float]
            search method for local_env for each atom.
        refine:str
            sort method for neighbors of each atom. The key in ``bond.refine_method``
        refined_strategy_param:dict
            parameters for refine
        numerical_tol:float
            numerical_tol
        pbc:list
            periodicity in 3 direction
            3-length list,each one is 1 or 0. such as [0,0,0],The 1 mean in this direction is with periodicity.
        """

        super().__init__(n_jobs=1, on_errors='raise', return_type='any')
        self.nn_strategy = nn_strategy
        if refine is None:
            self.refine = universe_refine
        else:
            self.refine = refine_method[refine]
        self.refined_strategy_param = refined_strategy_param if isinstance(refined_strategy_param, dict) else {}
        self.cutoff = self.nn_strategy if isinstance(self.nn_strategy, float) else None
        self.numerical_tol = numerical_tol
        self.pbc = pbc

    def convert(self, structure: StructureOrMolecule):
        """

        Parameters
        ----------
        structure:Structure
            pymatgen Structure

        Returns
        -------
        center_indices:np.ndarray of shape(n,)
            center indexes
        neighbor_indices:np.ndarray of shape(n,fill_size)
            neighbor_indexes for each center_index
            `fill_size` is the parameter of `refine` function
        images:np.ndarray of shape(n,3)
            offset vector in 3 orientations.
        distances:np.ndarray of shape(n,fill_size)
            distance of neighbor_indexes for each center_index
        """

        if isinstance(self.nn_strategy, float):
            return self.get_graphs_within_cutoff_merge_sort(structure)
        else:
            return self.get_graphs_strategy_merge_sort(structure)

    def get_graphs_within_cutoff(self,
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
                                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For quick get bond distance"""
        cutoff = self.cutoff
        numerical_tol = self.numerical_tol
        center_indices, neighbor_indices, images, distances = self.get_graphs_within_cutoff(structure, cutoff,
                                                                                            numerical_tol)

        center_indices, neighbor_indices, images, distances = self.refine(center_indices, neighbor_indices, distances,
                                                                          images,
                                                                          **self.refined_strategy_param)

        if len(center_indices.tolist()) == len(structure.species):
            return center_indices, neighbor_indices, images, distances
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), cutoff, cutoff + 2)
            )
            center_indices, neighbor_indices, images, distances = self.get_graphs_within_cutoff_merge_sort(structure)
            return center_indices, neighbor_indices, images, distances

    def get_graphs_strategy_merge_sort(self, structure: StructureOrMolecule
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For get bond distance with different strategy, for different nn_staagy could be rewrite"""
        assert hasattr(self.nn_strategy, "cutoff")

        index1 = []
        index2 = []
        bonds = []
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            index2i = [neighbor["site_index"] for neighbor in neighbors]
            bondsi = [neighbor["weight"] for neighbor in neighbors]
            index2.extend(index2i)
            bonds.extend(bondsi)

        center_indices, neighbor_indices, images, distances = self.refine(np.array(index1), np.array(index2),
                                                                          np.array(bonds),
                                                                          **self.refined_strategy_param)

        if len(center_indices.tolist()) == len(structure.species):
            return center_indices, neighbor_indices, images, distances
        else:
            print("For {}, There is no neighbor in cutoff {} A, try with cutoff {} A for this structure.".format(
                str(structure.composition), self.nn_strategy.cutoff, self.nn_strategy.cutoff + 2)
            )
            self.nn_strategy.cutoff += 2
            center_indices, neighbor_indices, images, distances = self.get_graphs_strategy_merge_sort(structure)

            return center_indices, neighbor_indices, images, distances


# ###############bond##########################################

class BondGaussianConverter(BaseFeature):
    """
    For bond distance.
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = None, width=0.5):
        """

        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        if centers is None:
            centers = np.linspace(0, 5, 100)
        self.centers = centers
        self.width = width
        self.d2 = False
        super(BondGaussianConverter, self).__init__()

    def _convert(self, d: np.ndarray) -> np.ndarray:
        """
        expand distance vector d with given parameters

        Args:
            d: (1d array) distance array

        Returns:
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)

        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)

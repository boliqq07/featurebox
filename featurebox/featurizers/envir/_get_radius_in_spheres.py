from typing import Tuple

import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from featurebox.utils.predefined_typing import StructureOrMolecule
from utils.general import re_pbc


def get_radius_in_spheres(
        structure: StructureOrMolecule, nn_strategy=None, cutoff: float = 5.0,
        numerical_tol: float = 1e-8,
        pbc=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff.

    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance
        nn_strategy(str):not used
    Returns:
        center_indices, neighbor_indices, images, distances, center_prop


    """
    _ = nn_strategy

    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        if pbc is not False:
            pbc = re_pbc(pbc, return_type="int")
        else:
            pbc = np.array([0, 0, 0])
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0])
    else:
        raise ValueError("structure type not supported")

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
    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], \
           distances[exclude_self], None

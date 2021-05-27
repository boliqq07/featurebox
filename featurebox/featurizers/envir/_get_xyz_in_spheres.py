import itertools
import math
from typing import Union, List

import numpy as np
from mgetool.tool import tt
from pymatgen.core import Structure


def get_xyz_in_spheres(
        all_coords: np.ndarray,
        reciprocal_lattice_abc,
        matrix,
        inv_matrix,
        cutoff: float,
        pbc: Union[bool, List[bool]] = True,
        numerical_tol: float = 1e-8,

):
    """
    For each point in `center_coords`, get all the neighboring points in `all_coords` that are within the
    cutoff radius `r`.

    Args:
        all_coords: (list of cartesian coordinates) all available points
        cutoff: (float) cutoff radius
        pbc: (bool or a list of bool) whether to set periodic boundaries
        numerical_tol: (float) numerical tolerance
        matrix:lattice.matrix
        inv_matrix；lattice.inv_matrix
        reciprocal_lattice_abc:lattice.reciprocal_lattice.abc

    """
    if isinstance(pbc, bool):
        pbc = [pbc] * 3
    center_coords = all_coords
    pbc = np.array(pbc, dtype=np.bool_)  # type: ignore
    center_coords_min = np.min(center_coords, axis=0)
    center_coords_max = np.max(center_coords, axis=0)
    # The lower bound of all considered atom coords
    global_min = center_coords_min - cutoff - numerical_tol
    global_max = center_coords_max + cutoff + numerical_tol
    ind = np.arange(len(all_coords))
    if np.any(pbc):
        recp_len = np.array(reciprocal_lattice_abc)
        maxr = np.ceil((cutoff + 0.15) * recp_len / (2 * math.pi))
        frac_coords = np.dot(center_coords, inv_matrix)
        nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
        nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
        nmin = np.zeros_like(nmin_temp)
        nmin[pbc] = nmin_temp[pbc]
        nmax = np.ones_like(nmax_temp)
        nmax[pbc] = nmax_temp[pbc]
        all_ranges = [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax)]

        # temporarily hold the fractional coordinates
        image_offsets = np.dot(all_coords, inv_matrix)
        all_fcoords = []
        # only wrap periodic boundary
        for k in range(3):
            if pbc[k]:  # type: ignore
                all_fcoords.append(np.mod(image_offsets[:, k: k + 1], 1))
            else:
                all_fcoords.append(image_offsets[:, k: k + 1])
        all_fcoords = np.concatenate(all_fcoords, axis=1)
        image_offsets = image_offsets - all_fcoords
        coords_in_cell = np.dot(all_fcoords, matrix)
        # Filter out those beyond max range
        valid_coords = []
        valid_images = []
        valid_indices = []

        for image in itertools.product(*all_ranges):
            coords = np.dot(image, matrix) + coords_in_cell
            valid_index_bool = np.all(
                np.bitwise_and(coords > global_min[None, :], coords < global_max[None, :]),
                axis=1,
            )
            if np.any(valid_index_bool):
                valid_coords.append(coords[valid_index_bool])
                valid_images.append(np.tile(image, [np.sum(valid_index_bool), 1]) - image_offsets[valid_index_bool])
                valid_indices.extend([k for k in ind if valid_index_bool[k]])
        if len(valid_coords) < 1:
            return [[]] * len(center_coords)
        valid_coords = np.concatenate(valid_coords, axis=0)
        valid_images = np.concatenate(valid_images, axis=0)
        valid_indices = np.array(valid_indices)

    else:
        valid_coords = all_coords  # type: ignore
        valid_images = np.array([[0, 0, 0]] * len(valid_coords))
        valid_indices = np.arange(len(valid_coords))

    center=[]
    neighbor =[]
    images=[]
    coords= []
    lc = center_coords.shape[0]
    lv = valid_coords.shape[0]
    for i in range(lc):
        ci = center_coords[i]
        v = valid_coords - ci
        rs = (v[:,0]**2+v[:,1]**2+v[:,2]**2)**0.5
        index = rs <= cutoff

        center.append(np.full(lv,i)[index])
        neighbor.append(valid_indices[index])
        images.append(valid_images[index,:])
        coords.append(v[index,:])

    center = np.concatenate(center,axis=0)
    neighbor = np.concatenate(neighbor,axis=0)
    images = np.concatenate(images,axis=0)
    coords = np.concatenate(coords,axis=0)
    coords = np.concatenate((((coords[:,0]**2+coords[:,1]**2+coords[:,2]**2)**0.5).reshape(-1,1),coords),axis=1)

    return center,neighbor,images,coords


# structure = Structure.from_file("S2-CONTCAR")
# tt.t
# get_points_ = get_xyz_inspheres(structure.cart_coords,
#                                     reciprocal_lattice_abc=structure.lattice.reciprocal_lattice.abc,
#                                     matrix=structure.lattice.matrix,
#                                     inv_matrix=structure.lattice.inv_matrix,
#                                     r=5.0, pbc=True,
#                                     )
# tt.t#tt.t
# get_points_ = get_xyz_inspheres(structure.cart_coords,
#                                     reciprocal_lattice_abc=structure.lattice.reciprocal_lattice.abc,
#                                     matrix=structure.lattice.matrix,
#                                     inv_matrix=structure.lattice.inv_matrix,
#                                     r=5.0, pbc=True,
#                                     )
# tt.t#
# tt.p

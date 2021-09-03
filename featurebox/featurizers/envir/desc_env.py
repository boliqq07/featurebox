"""Use descriptors form ``pyXtal_FF``, in :mod:`featurebox.test_featurizers.descriptors`
"""
import copy
from typing import Dict, Tuple

import numpy as np
from pymatgen.core import Structure, Molecule

from featurebox.featurizers.descriptors.ACSF import ACSF
from featurebox.featurizers.descriptors.EAD import EAD
from featurebox.featurizers.descriptors.EAMD import EAMD
from featurebox.featurizers.descriptors.SO3 import SO3
from featurebox.featurizers.descriptors.SO4 import SO4_Bispectrum
from featurebox.featurizers.descriptors.SOAP import SOAP
from featurebox.featurizers.descriptors.behlerparrinello import BehlerParrinello
from featurebox.featurizers.descriptors.wACSF import wACSF
from featurebox.featurizers.envir._get_xyz_in_spheres import get_xyz_in_spheres
from featurebox.utils.general import aaa
from featurebox.utils.look_json import mark_classes

DesDict = mark_classes([
    ACSF,
    BehlerParrinello,
    EAD,
    EAMD,
    SOAP,
    SO3,
    SO4_Bispectrum,
    wACSF,
])

for i, j in DesDict.items():
    locals()[i] = j


def get_strategy2_in_spheres(structure, nn_strategy, cutoff, numerical_tol=None, cutoff_name="cutoff", pbc=False):
    _ = numerical_tol
    if isinstance(structure, Structure) or isinstance(structure, Molecule):
        atoms = aaa.get_atoms(structure)
    else:
        atoms = structure

    nn_strategy_ = copy.copy(nn_strategy)

    setattr(nn_strategy_, cutoff_name, cutoff)

    result = nn_strategy_.calculate(atoms)

    _, _, _, _, b5 = get_5_result(result)

    a1, a2, a3, a4, _ = get_xyz_in_spheres(structure, nn_strategy=None, cutoff=cutoff, numerical_tol=numerical_tol,
                                           pbc=pbc,
                                           )

    return a1, a2, a3, a4, b5


def get_5_result(d: Dict, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    Args:
        d: dict, dict of descriptor. at lest contain "x" and "dxdr"
        fill_size: int, unstable.

    Returns:
        (center_indices,center_prop, neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size), None.
        center_prop: np.ndarray 1d(N,l_c).\n

    """
    center, seq = d["x"], d.get("seq")
    atom_len = center.shape[0]

    center_indices = np.array(range(atom_len))

    return center_indices, np.array(None), np.array(None), np.array(None), center

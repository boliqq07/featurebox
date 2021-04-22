"""Define several typing for convenient use"""

from typing import Union, Callable, Optional, Any, List, Tuple

import numpy as np
from ase import Atoms
from pymatgen.core import Structure, Molecule

OptStrOrCallable = Optional[Union[str, Callable[..., Any]]]
StructureOrMolecule = Union[Structure, Molecule]
StructureOrMoleculeOrAtoms = Union[Structure, Molecule, Atoms]
VectorLike = Union[List[float], np.ndarray]
ListTuple = Union[List, Tuple]

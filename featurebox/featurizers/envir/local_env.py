"""Use NearNeighbors from pymatgen."""
from typing import Dict, List

from pymatgen.core import Element
from pymatgen.analysis.local_env import (
    NearNeighbors,
    VoronoiNN,
    MinimumDistanceNN,
    MinimumVIRENN,
    BrunnerNN_reciprocal,
    BrunnerNN_real,
    BrunnerNN_relative,
    EconNN,
    CrystalNN,
)
from pymatgen.core import Molecule
from pymatgen.core.structure import Structure

from featurebox.utils.look_json import mark_classes


def _is_in_targets(site, targets):
    """
    Test whether a site contains elements in the target list

    Args:
        site (Site): Site to assess
        targets ([Element]) List of elements
    Returns:
         (boolean) Whether this site contains a certain list of elements
    """
    elems = _get_elements(site)
    for elem in elems:
        if elem not in targets:
            return False
    return True


def _get_elements(site):
    """
    Get the list of elements for a Site

    Args:
         site (Site): Site to assess
    Returns:
        [Element]: List of elements
    """
    try:
        if isinstance(site.specie, Element):
            return [site.specie]
        return [Element(site.specie)]
    except Exception:
        return site.species.elements


class MinimumDistanceNNAll(NearNeighbors):
    """
    Determine bonded sites by fixed cutoff.
    """

    def __init__(self, cutoff: float = 4.0):
        """
        Args:
            cutoff (float): cutoff radius in Angstrom to look for trial
                near-neighbor sites (default: 4.0).
        """
        self.cutoff = cutoff

    def get_nn_info(self, structure: Structure, n: int) -> List[Dict]:
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the closest neighbor
        distance-based method.

        Args:
            structure (Structure): input structure.
            n (integer): index of site for which to determine near
                neighbors.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a neighbor site, its image location,
                and its weight.
        """

        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)

        siw = []
        for nn in neighs_dists:
            siw.append(
                {
                    "site": nn,
                    "image": self._get_image(structure, nn),
                    "weight": nn.nn_distance,
                    "site_idx": self._get_original_site(structure, nn),
                }
            )
        return siw


class AllAtomPairs(NearNeighbors):
    """
    Get all combinations of atoms as bonds in a molecule
    """

    def get_nn_info(self, molecule: Molecule, n: int) -> List[Dict]:
        """
        Get near neighbor information
        Args:
            molecule (Molecule): pymatgen Molecule
            n (int): number of molecule

        Returns: List of neighbor dictionary

        """
        site = molecule[n]
        siw = []
        for i, s in enumerate(molecule):
            if i != n:
                siw.append({"site": s, "image": None, "weight": site.distance(s), "site_idx": i})
        return siw


class UserVoronoiNN(VoronoiNN):

    def _extract_nn_info(self, structure, nns):
        """Given Voronoi NNs, extract the NN info in the form needed by NearestNeighbors

        Args:
            structure (Structure): Structure being evaluated
            nns ([dicts]): Nearest neighbor information for a structure
        Returns:
            (list of tuples (Site, array, float)): See nn_info
        """

        # Get the target information
        if self.targets is None:
            targets = structure.composition.elements
        else:
            targets = self.targets

        # Extract the NN info
        siw = []
        max_weight = max(nn[self.weight] for nn in nns.values())
        for nstats in nns.values():
            site = nstats["site"]
            if nstats[self.weight] > self.tol * max_weight and _is_in_targets(
                    site, targets
            ):
                nn_info = {
                    "site": site,
                    "image": self._get_image(structure, site),
                    "weight": nstats[self.weight] / max_weight,
                    "site_index": self._get_original_site(structure, site),
                }

                if self.extra_nn_info:
                    # Add all the information about the site
                    poly_info = nstats
                    del poly_info["site"]
                    nn_info["poly_info"] = poly_info
                    nn_info["add_image"] = (poly_info["solid_angle"], poly_info["volume"], poly_info["face_dist"],
                                            poly_info["area"], poly_info["n_verts"])
                siw.append(nn_info)
        return siw


NNDict = mark_classes([
    VoronoiNN,
    UserVoronoiNN,
    MinimumDistanceNN,
    MinimumDistanceNNAll,
    AllAtomPairs,
    # OpenBabelNN,  #BabelMolAdaptor requires openbabel to be installed with Python bindings.
    # CovalentBondNN, just for molecule
    MinimumVIRENN,
    # MinimumOKeeffeNN, # just for specific element/
    BrunnerNN_reciprocal,
    BrunnerNN_real,
    BrunnerNN_relative,
    EconNN,
    CrystalNN,
    # CutOffDictNN, # just for specific element/
    # Critic2NN, #Critic2Caller requires the executable critic to be in the path.
    # Please follow the instructions at https://github.com/aoterodelaroza/critic2.
])

for i, j in NNDict.items():
    locals()[i] = j

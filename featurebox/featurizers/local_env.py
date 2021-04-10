import json
import os
import types
from inspect import getfullargspec
from typing import Dict, List, Union

from monty.json import MontyDecoder, _load_redirect
from pymatgen.analysis.local_env import (
    NearNeighbors,
    VoronoiNN,
    JmolNN,
    MinimumDistanceNN,
    OpenBabelNN,
    CovalentBondNN,
    MinimumVIRENN,
    MinimumOKeeffeNN,
    BrunnerNN_reciprocal,
    BrunnerNN_real,
    BrunnerNN_relative,
    EconNN,
    CrystalNN,
    CutOffDictNN,
    Critic2NN,
)
from pymatgen.core import Molecule
from pymatgen.core.structure import Structure

REDIRECT = _load_redirect(
    os.path.join(os.path.expanduser("~"), ".monty.yaml"))


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


def as_dict(self) -> dict:
    """
    A JSON serializable dict representation of an object.
    """
    d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__, "@version": None}

    dk = {}
    spec = getfullargspec(self.__init__)
    args = spec.args

    def recursive_as_dict(obj):
        if isinstance(obj, (list, tuple)):
            return [recursive_as_dict(it) for it in obj]
        if isinstance(obj, dict):
            return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        return obj

    for c in args:
        if c != "self":
            try:
                a = self.__getattribute__(c)
            except AttributeError:
                try:
                    a = self.__getattribute__("_" + c)
                except AttributeError:
                    raise NotImplementedError(
                        "Unable to automatically determine as_dict "
                        "format from class. MSONAble requires all "
                        "args to be present as either self.argname or "
                        "self._argname, and kwargs to be present under"
                        "a self.kwargs variable to automatically "
                        "determine the dict format. Alternatively, "
                        "you can implement both as_dict and from_dict.")
            dk[c] = recursive_as_dict(a)
    if hasattr(self, "kwargs"):
        # type: ignore
        dk.update(**getattr(self, "kwargs"))  # pylint: disable=E1101
    if hasattr(self, "_kwargs"):
        dk.update(**getattr(self, "_kwargs"))  # pylint: disable=E1101

    d["all_args_kwargs"] = dk
    d["tp_na"] = self.__class__.__name__
    return d


def from_dict(d):
    """
    :param d: Dict representation.
    :return: MSONable class.
    """
    decoded = {
        k: MontyDecoder().process_decoded(v)
        for k, v in d.items() if not k.startswith("@")
    }

    if "tp_na" in decoded:
        cla = NNDict.get(decoded["tp_na"])
        if decoded["all_args_kwargs"] == {}:
            return cla()
        else:
            return cla(**decoded["all_args_kwargs"])
    else:
        raise TypeError


def to_json(self) -> str:
    """
    Returns a json string representation of the MSONable object.
    """
    return json.dumps(self, cls=UserMontyEncoder)


NNDict = {
    i.__name__ + "_D": type(i.__name__ + "_D", (i,),
                            {"as_dict": as_dict, "to_json": to_json, "from_dict": from_dict, "tp_na": True})
    for i in [
        NearNeighbors,
        VoronoiNN,
        JmolNN,
        MinimumDistanceNN,
        OpenBabelNN,
        CovalentBondNN,
        MinimumVIRENN,
        MinimumOKeeffeNN,
        BrunnerNN_reciprocal,
        BrunnerNN_real,
        BrunnerNN_relative,
        EconNN,
        CrystalNN,
        CutOffDictNN,
        Critic2NN,
        MinimumDistanceNNAll,
        AllAtomPairs,
    ]
}

for i, j in NNDict.items():
    locals()[i] = j


class UserMontyEncoder(json.JSONEncoder):
    """
    A Json Encoder which supports the MSONable API, plus adds support for
    numpy arrays, datetime objects, bson ObjectIds (requires bson).

    Usage::

        # Add it as a *cls* keyword when using json.dump
        json.dumps(object, cls=MontyEncoder)
    """

    def default(self, o) -> dict:  # pylint: disable=E0202
        try:
            d = o.as_dict()
            if "@module" not in d:
                d["@module"] = u"{}".format(o.__class__.__module__)
            if "@class" not in d:
                d["@class"] = u"{}".format(o.__class__.__name__)
            if "@version" not in d:
                d["@version"] = None
            return d
        except AttributeError:
            raise TypeError


def serialize(neighbors: NearNeighbors):
    """Add method dynamically ."""
    neighbors.as_dict = types.MethodType(as_dict, neighbors)
    neighbors.to_json = types.MethodType(to_json, neighbors)
    neighbors.from_dict = from_dict
    neighbors.tp_na = True
    return neighbors


def get_nn_strategy(nn_strategy: Union[str, NearNeighbors]):
    """Add method dynamically."""
    if isinstance(nn_strategy, str):
        if "_D" in nn_strategy:
            Nei = NNDict[nn_strategy]()
        else:
            Nei = NNDict[nn_strategy + "_D"]()
    elif isinstance(nn_strategy, NearNeighbors) or issubclass(nn_strategy, NearNeighbors):
        if not isinstance(nn_strategy, NearNeighbors):
            nn_strategy = nn_strategy()
        if nn_strategy.__class__.__name__ in NNDict:
            Nei = nn_strategy
        else:
            if nn_strategy.__class__.__name__ + "_D" in NNDict:
                Nei = NNDict[nn_strategy.__class__.__name__ + "_D"]()
                Nei.__dict__.update(nn_strategy.__dict__)
            else:
                raise TypeError("only accept str or object inherit from pymatgens.local_env.Neighbors")
    else:
        raise TypeError("only accept str or object inherit from pymatgens.local_env.Neighbors")
    return Nei

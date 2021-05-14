from collections import Callable

from featurebox.featurizers.atom.mapper import _StructurePymatgenPropMap


# re-site for the nature of the classification.
class StructurePymatgenPropMap(_StructurePymatgenPropMap):
    """
    Get property of pymatgen structure preprocessing.
    default ["density", "volume", "ntypesp"]

    Examples
    -----------
    >>> tmps = StructurePymatgenPropMap()
    >>> tmps.fit_transform()

    """

    def __init__(self, prop_name=None, func: Callable = None, return_type="df", **kwargs):
        """
        Args:
            prop_name:(str,list of str)
                prop name or list of prop name
                default ["density", "volume", "ntypesp"]
            func:(callable or list of callable)
                please make sure the size of it is the same with prop_name.
        """
        super(StructurePymatgenPropMap, self).__init__(prop_name, func, return_type, **kwargs)

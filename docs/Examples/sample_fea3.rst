Pymatgen Data
=============

The data are using the inner periodical_data.json in pymatgen elemental data.
::

    >>> tmps = AtomPymatgenPropMap(search_tp="name")
    >>> s = [{"H": 2, }, {"Po": 1}, {"C": 2}] # [i.species.as_dict() for i in pymatgen.structure.sites]


In addition, we could get structure state data by structure.
::

    >>> tmps = StructurePymatgenPropMap(prop_name = ["density", "volume", "ntypesp"])
    >>> a2 = tmps.convert(structurei)

This class is for structure but for atoms.

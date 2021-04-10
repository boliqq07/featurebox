Table data
================

Read table data as following format, and organize by composition.

===== ===== ===== =====
Data    F0    F1    ...
----- ----- ----- -----
H     V     V     ...
He    V     V     ...
Li    V     V     ...
Be    V     V     ...
...   ...   ...   ...
===== ===== ===== =====

Then:
::

    tmps = AtomTableMap(search_tp="name")
    s = [{"H": 2, }, {"Po": 1}, {"C": 2}]
    # [i.species.as_dict() for i in pymatgen.structure.sites]
    a = tmps.convert(s)

.. image:: 2_1.png

In default, the proportion would be multiplied in data, also you can neglect weight.
::

    tmps = AtomTableMap(search_tp="name", weight=False)
    s = [{"H": 2, }, {"Po": 1}, {"C": 2}]
    # [i.species.as_dict() for i in pymatgen.structure.sites]
    a2 = tmps.convert(s)

.. image:: 2_2.png

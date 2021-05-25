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

Then run the code.
::

    >>> tmps = AtomTableMap(search_tp="name")
    >>> com = [{"H": 2, }, {"Po": 1}, {"C": 2}]
    >>> a = tmps.convert(com)

.. image:: 2_1.png

In default, the proportion would be multiplied in data, also you can neglect weight.
::

    >>> tmps = AtomTableMap(search_tp="name", weight=False)
    >>> com = [{"H": 2, }, {"Po": 1}, {"C": 2}]
    >>> a2 = tmps.convert(com)

.. image:: 2_2.png


Note
::

    com = [i.species.as_dict() for i in pymatgen.structure.sites]

    or

    com =  [{i.element.symbol:1} for i in structure.species]
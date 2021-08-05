Json Data
==============

The key json must be element name, such as {"H": ... ,"He": ... }, and The structures is pymatgen ``Structure`` list.

Index by structure
::

>>> from featurebox.featurizers.atom.mapper import AtomJsonMap
>>> tmps = AtomJsonMap(search_tp="number", embedding_dict="elemental_MEGNet.json")
>>> a = tmps.convert(structures)

The return data are properties of 1, 76 elements.

Index by number, with your-self json
::

>>> from featurebox.featurizers.atom.mapper import AtomJsonMap
>>> tmps = AtomJsonMap(search_tp="number",embedding_dict="elemental_MEGNet.json")
>>> s = [1,76]
>>> # could from [i.specie.Z for i in structure]
>>> a = tmps.convert(s)

The return data are properties of 1, 76 elements.

.. image:: 1_1.png

Index by dict data
::

>>> from featurebox.featurizers.atom.mapper import AtomJsonMap
>>> tmps = AtomJsonMap(search_tp="name")
>>> s = [{"H": 2, }, {"Al": 1}]
>>> # could from [i.species.as_dict() for i in pymatgen.structure.sites]
>>> or [{i.element.symbol:1} for i in structure.species]
>>> a = tmps.convert(s)

.. image:: 1_3.png

Batch data
::

>>> from featurebox.featurizers.atom.mapper import AtomJsonMap
>>> tmps = AtomJsonMap(search_tp="name")
>>> s = [[{"H": 2, }, {"Ce": 1}],[{"H": 2, }, {"Al": 1}]]
>>> a = tmps.transform(s)

The return data are list of 2 np.ndarray data.

.. image:: 1_2.png

.. image:: 1_3.png
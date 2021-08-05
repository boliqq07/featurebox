Batch Transform Data
======================

If you don't have a preference for features or idea, just try BatchFeature,
we using features from pymatgen.

1.Transform structure list.

>>> from featurebox.featurizers.batch_feature import BatchFeature
>>> bf = BatchFeature(data_type="structures", return_type="df")
>>> data = bf.fit_transform(structures_list)

``structures_list`` is list of ``struceture`` of ``pymatgen``.

.. image:: structures0.gif

2.Transform composition list.

>>> from featurebox.featurizers.batch_feature import BatchFeature
>>> bf = BatchFeature(data_type="composition")
>>> com = [[{str(i.symbol): 1} for i in structure.species]  for structure in sturctures]
>>> #where com is element list
>>> data = bf.fit_transform(com)

.. image:: composition0.gif

3.Transform element list.

>>> from featurebox.featurizers.batch_feature import BatchFeature
>>> bf = BatchFeature(data_type="elements")
>>> aa =[]
>>> aas = [[{str(i.symbol): 1} for i in structure.species] for structure in sturctures]
>>> [aa.extend(i) for i in aas]
>>> #where aa is element list
>>> data = bf.fit_transform([aa])

Note
::

    It is highly recommended that using this function as a beginner,
    Because we can customize more and more powerful converters.

Just go on !

Batch Transform Data
======================

If you don't have a preference for features or idea, just try BatchFeature,
we using features from pymatgen.

Transform structure list.
::

    sppm = BatchFeature(data_type="structures", return_type="df")
    data = sppm.fit_transform(structures_list)

``structures_list`` is list of ``struceture`` of ``pymatgen``.

.. image:: structures0.gif

Transform composition list.
::

    composition = [i.composition.to_reduced_dict for i in structure_list]
    data = sppm.fit_transform(composition)

.. image:: composition0.gif

Transform element list.
::

    sppm = BatchFeature(data_type="elements")
    aa =[]
    aas = [[i.species.as_dict() for i in structure.sites] for structure in sturctures]
    [aa.extend(i) for i in aas]
    data = sppm.fit_transform([aa])

Note
::

    It is highly recommended that using this function as a beginner,
    Because we can customize more and more powerful converters.

Just go on !

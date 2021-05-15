Structure Graph Processing
==========================

Get structure data::

    >>> data = pd.read_pickle("pymatgen_structures_list.pkl_pd")
    >>> y = ...
    >>> # using your self pymatgen.structure List

Check structure elements in scope::

    >>> ce = CheckElements.from_pymatgen_structures()
    >>> checked_data = ce.check(data)
    >>> y = np.array(y)[ce.passed_idx()]

Transform parallel::

    >>> gt = CrystalBgGraph(n_jobs=4)
    >>> in_data = gt.transform(checked_data)
    >>> pd.to_pickle((in_data, y), "in_data_no_sgt.pkl_pd")
    >>> """you can save the in_data to the local disk to prevent double counting."""

Using data (extension)::

    >>> gen = GraphGenerator(*data, targets=y)
    >>> loader = MGEDataLoader(
    ...                    dataset=gen,
    ...                    batch_size=1,
    ...                    shuffle=True,
    ...                    num_workers=0,
    ...                    )

Output Data:

=====================  ====
 Data                  y
---------------------  ----
 s1:[[...],...,[...]]  y1
 s1:[[...],...,[...]]
 s1:[[...],...,[...]]
 s2:[[...],...,[...]]  y2
 s2:[[...],...,[...]]
 s3...                 y3
 s3...
 s3...
 s4:[[...],...,[...]]  y4
 s4:[[...],...,[...]]
=====================  ====

Note::

    The size of transformed data and y are different. but the first of graph is "node_atom_idx" to appoint the sample index.
    Thus, in network, there must be one step to stack the data with same "node_atom_idx"
    This is the critical point of graph network, such as CGCNN, and MGENet and so on!!!!!
    The "node_atom_idx" are add in GraphGenerator.


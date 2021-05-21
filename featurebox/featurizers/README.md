Structure Graph Processing
--------------------------

Get structure data
------------------
    >>> data = pd.read_pickle("pymatgen_structures_list.pkl_pd")
    >>> y = ...
    >>> # using your self pymatgen.structure List

Check structure elements in scope
---------------------------------
    >>> ce = CheckElements.from_pymatgen_structures()
    >>> checked_data = ce.check(data)
    >>> y = np.array(y)[ce.passed_idx()]

Transform parallel
------------------
    >>> gt = CrystalBgGraph(n_jobs=2)
    >>> in_data = gt.transform(checked_data)
    >>> """you can save the in_data to the local disk to prevent double counting."""

Add your own data*
------------------
See Note.

Using data
----------

    >>> gen = GraphGenerator(*data, targets=y)
    >>> loader = MGEDataLoader(
                        dataset=gen,  
                        batch_size=1,  
                        shuffle=True,  
                        num_workers=0,  
                        )

=====================  ====
 data                  y
---------------------  ----
 s1:[[1],...,[a]]      y1
 s1:[[1],...,[o]]
 s1:[[1],...,[e]]
 s2:[[2],...,[s]]      y2
 s2:[[2],...,[k]]         
 s3...                 y3
 s3...        
 s3...       
 s4:[[4],...,[f]]      y4
 s4:[[4],...,[v]]         
=====================  ====

Note
----
    The size of transformed data and y are different. but the first of graph is "node_atom_idx" to appoint the sample index.
    Thus, in network, there must be one step to stack the data with same "node_atom_idx"
    This is the critical point of graph network, such as CGCNN, and MGENet and so on!!!!!
    The "node_atom_idx" and so on are add in GraphGenerator.

Note
----
    To add your own data, please add the atom feature, compound feature to  ``atom_fea`` and ``state_fea``
    to the last axis, ``nbr_fea`` similarly.  
    But for bond feature, we adjust re-write the bond_generator ranther than add there directly 
    unless you are clear and keep ``atom_nbr_idx`` consistent with the bond fea data you added.

    ``atom_fea``: list of np.ndarray, shape (N, atom_fea_len)
       center properties.
    ``nbr_fea``: list of np.ndarray, shape (N, fill_size, atom_fea_len).
       neighbor_indexes for each center_index.
       `fill_size` default = 5.
    ``state_fea``: list of np.ndarray, shape (state_fea_len,)
        state feature.
    ``atom_nbr_idx``: list of np.ndarray, shape (N, fill_size)
       neighbor for each center, fill_size default is 5.
1.Structure Graph Processing
==============================

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

    >>> gt = StructureGraphGEO(n_jobs=2)
    >>>
    >>> """dict data for show."""
    >>> in_data = gt.transform(checked_data) 

    >>> """1. torch_geometric.data.Data type data."""
    >>> data = gt.transform_and_to_data(checked_data)

    >>> """2. torch_geometric.data.Data type data and save to local path."""
    >>> data = gt.transform_and_save(checked_data,root_dir = "path")

Using data
----------

    >>> from torch_geometric.data import DataLoader
    >>> """1. Just use data (small data)."""
    >>> loader = DataLoader(
                        dataset=data,  
                        batch_size=1,  
                        shuffle=True,  
                        num_workers=0,  
                        )

    >>> from torch_geometric.data import DataLoader
    >>> """2. Use local data (middle data)."""
    >>> gen = InMemoryDatasetGeo(root="path")
    >>> loader = DataLoader(
                        dataset=gen,  
                        batch_size=1,  
                        shuffle=True,  
                        num_workers=0,  
                        )

    >>> from torch_geometric.data import DataLoader
    >>> """3. Use local data (large data)."""
    >>> gen = DatasetGeo(root="path")
    >>> loader = DataLoader(
                        dataset=gen,  
                        batch_size=1,  
                        shuffle=True,  
                        num_workers=0,  
                        )

Note
----

    Each Graph data (for each structure):

    ``x``: Node feature matrix. np.ndarray, with shape [num_nodes, num_node_features]
    
    ``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long
    
    ``edge_attr``: Edge feature matrix. np.ndarray, with shape [num_edges, num_edge_features]
    
    ``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]
    
    ``y``: target. np.ndarray, shape (1, num_target), default shape (1,)
    
    ``state_attr``: state feature. np.ndarray, shape (1, num_state_features)
    
    ``z``: atom numbers. np.ndarray, with shape [num_nodes,]
    
    Where the state_attr is added newly.

2.Old Version(Structure Graph Processing) Deprecated
=====================================================

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

===================== ==== data y
---------------------  ----
s1:[[1],...,[a]]      y1 s1:[[1],...,[o]]
s1:[[1],...,[e]]
s2:[[2],...,[s]]      y2 s2:[[2],...,[k]]         
s3... y3 s3...        
s3...       
s4:[[4],...,[f]]      y4 s4:[[4],...,[v]]         
===================== ====

Note
----

    This part is deprecated.

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
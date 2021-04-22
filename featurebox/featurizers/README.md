## Structure Graph Processing

#### Preprocessing

###### Get structure data

    data = pd.read_pickle("pymatgen_structures_list.pkl_pd")
    y = ...
    # using your self pymatgen.structure List

###### Check structure elements in scope

    ce = CheckElements.from_pymatgen_structures()
    checked_data = ce.check(data)
    y = np.array(y)[ce.passed_idx()]

###### Transform parallel

    gt = GraphsTool(graph_object=CrystalBgGraph, n_jobs=2, 
                    batch_calculate=True, batch_size=10)
                    
    in_data = gt.transform(checked_data)
    
    """you can save the in_data to the local disk to prevent double counting."""

#### Using data

    gen = GraphGenerator(*data, targets=y)
    loader = MGEDataLoader(
                        dataset=gen,  
                        batch_size=1,  
                        shuffle=True,  
                        num_workers=0,  
                        )

#### Note

The size of transformed data and y are different. but the first of graph is "node_atom_idx" to appoint the sample index.

Thus, in network, there must be one step to stack the data with same "node_atom_idx"
This is the critical point of graph network, such as CGCNN, and MGENet and so on!!!!!

The "node_atom_idx" are add in GraphGenerator.

     in_data               y
     
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

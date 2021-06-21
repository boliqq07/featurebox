__all__ = ["cgcnn", "flow", "megnet", "symnet", "models_geo"]

__doc__ = """

Input Note/ Read Me First
-------------------------

Each Graph data in (for each structure) in ``models_geo``:

``x``: Node feature matrix. np.ndarray, with shape (num_nodes, num_node_features)

``edge_index``: Graph connectivity in COO format. np.ndarray, with shape (2, num_edges) and type torch.long

``edge_attr``: Edge feature matrix. np.ndarray, with shape (num_edges, num_edge_features)

``pos``: Node position matrix. np.ndarray, with shape (num_nodes, num_dimensions)

``y``: target. np.ndarray, shape (1, num_target), default shape (1,)

``state_attr``: state feature. np.ndarray, shape (1, num_state_features)

``z``: atom numbers. np.ndarray, with shape (num_nodes,)

Where the state_attr is added newly.

.. note::

    For different model, not all attributes are used.
    
"""

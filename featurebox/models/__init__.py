__all__ = ["cgcnn", "flow", "megnet"]

__doc__ = """
Input Note/ Read Me First
-------------------------

There are 3 length for first dim(axis) of data.

`N_atom` > `N_ele`> `N_node`.

1. `N_node`: This is consistent with the number of target. `N_node`==`N_target`. In brief, The first dim(axis) of data
   should be reduced to `N_node` to calculated error in the final network.

   `N_node` is the min data number. which can be expand in to `N_atom`, and `N_ele`.

   Examples:

        new_data = self.expand_idx(data, node_atom_idx)
        new_data = self.expand_idx(data, node_ele_idx)

2. `N_ele`: This is the number of type of atomic number.

   This type is not used very much.

   `N_node` is the min data number. which can be expand in to and `N_atom` merged to `N_node`.

   Examples:

        new_data = self.expand_idx(data, ele_atom_idx)
        new_data = self.merge_idx(data, node_ele_idx)

3. `N_atom` (`N`): This is the number of all atom in batch data.

   The data with `N` could be merged in to `N_node` by node_atom_idx.

   Examples:

        new_data = self.merge_idx(data, node_atom_idx)
        new_data = self.merge_idx(data, ele_atom_idx)

Common data
--------------

`atom_fea`:(torch.Tensor) torch.float32, shape (N, atom_fea_len)

`nbr_fea`:(torch.Tensor) torch.float32, shape (N, fill_size, atom_fea_len) M default is 5.

`state_fea`: (torch.Tensor) torch.float32, shape (N_node, state_fea_len)

`atom_nbr_idx`: (torch.Tensor) torch.int64, shape (N, fill_size) ,fill_size default is 5.

`node_atom_idx`: (list of torch.Tensor) torch.int64, each one shape is different.

`node_ele_idx`: (list of torch.Tensor) torch.int64, each one shape is different.

`ele_atom_idx`: (list of torch.Tensor) torch.int64, each one shape is different.



"""

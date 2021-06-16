import os
import os.path as osp
from shutil import rmtree

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset


class InMemoryDatasetGeo(InMemoryDataset):
    """For small data <= 2000."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, re_process_init=True, load_mode="i"):
        """

        Args:
            load_mode (str): load from the independent data in different files: "i",
                load from the batch data in one overall file: "o".
            re_process_init (bool): process raw data or not. if there is no raw data but processed data is offered.
                this parameter could be False.
            root (string, optional): Root directory where the dataset should be
                saved. (optional: :obj:`None`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        """

        if load_mode in ["o", "O", "overall", "Overall"]:
            self._load = self._load_overall
            # "overall"
        elif load_mode in ["I", "R", "i", "r", "Respective", "respective"]:
            # "Respectively"
            self._load = self._load_respective
        else:
            raise NotImplementedError("load_mode = 'o' or 'i'")

        super(InMemoryDatasetGeo, self).__init__(root, transform, pre_transform, pre_filter=pre_filter)

        if re_process_init:
            self.re_process()
        else:
            self.data, self.slices = torch.load(osp.join(self.processed_paths[0]))

    def _load_overall(self):
        assert len(self.raw_paths) == 1, "There is more than one .pt file,and not sure which one to import."
        return [Data.from_dict(i) for i in torch.load(self.raw_paths[0])]

    def _load_respective(self):
        return [Data.from_dict(torch.load(i)) for i in self.raw_paths]

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = self._load()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def re_process(self):
        """Re-process for skip reset in 56 line in ``InMemoryDataset``
        `self.data, self.slices = None, None` ."""
        rmtree(self.processed_dir)

        os.makedirs(self.processed_dir)
        self.process()

        print('Done!')


class DatasetGEO(Dataset):
    """For very very huge data."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, re_process_init=False, load_mode="i"):
        """

        Args:
            load_mode (str): load from the independent data in different files: "i",
                load from the batch data in one overall file: "o".
            re_process_init (bool): process raw data or not. if there is no raw data but processed data is offered.
                this parameter could be False.
            root (string, optional): Root directory where the dataset should be
                saved. (optional: :obj:`None`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        """
        if load_mode not in ["I", "R", "i", "r", "Respective", "respective"]:
            raise NotImplementedError("Just accept mode 'i' for ``DatasetGEO``.")

        self.re_process_init = re_process_init

        super(DatasetGEO, self).__init__(root, transform, pre_transform, pre_filter=pre_filter)

        if re_process_init:
            self.re_process()
        else:
            pass

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @staticmethod
    def _files_exist(path):
        if os.path.isdir(path):
            files = os.listdir(path)
            return len(files) > 0
        else:
            return False

    @property
    def processed_file_names(self):
        if self._files_exist(self.processed_dir) and not self.re_process_init:
            return os.listdir(self.processed_dir)
        else:
            return ["data_{}.pt".format(i) for i in range(len(self.raw_file_names))]

    def re_process(self):
        """For temporary debug"""
        rmtree(self.processed_dir)
        os.makedirs(self.processed_dir)
        self.process()

        print('Done!')

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            try:
                data = Data.from_dict(torch.load(raw_path))
            except AttributeError as e:
                print(e)
                raise AttributeError("Just accept mode 'i' for ``DatasetGEO``, which load one at a time.",
                                     "The {} may be a batch of data. "
                                     "That is , if your raw data are save in one file overall, with 'o' mode."
                                     "please turn to ``InMemoryDatasetGeo``".format(raw_path))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

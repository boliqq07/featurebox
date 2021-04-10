import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.base_graph import CrystalGraph
from featurebox.featurizers.generator import GraphGenerator, MGEDataLoader


class Test_CrystalBgGraphDisordered(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_checked = ce.check(self.data)[:10]

        gt = CrystalGraph(n_jobs=1, batch_calculate=True, batch_size=10)
        data = gt.transform(self.data0_checked)

        gen = GraphGenerator(*data, targets=None)
        self.gen=gen
        from sklearn.base import BaseEstimator

    def test_sf(self):
        data0 = self.gen[0]

    def test_sf_data(self):
        gen = self.gen
        loader = MGEDataLoader(
            dataset=gen,  # torch TensorDataset format
            batch_size=3,  # 最新批数据
            shuffle=False,  # 是否随机打乱数据
            num_workers=0,  # 用于加载数据的子进程
            collate_marks = ('c', 'c', 's', 'c', 'f', 'c','f')
        )
        for k,i in enumerate(loader):
            # if k<5:
            print("\n")
            print(k)


if __name__ == '__main__':
    unittest.main()

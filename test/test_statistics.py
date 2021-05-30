import unittest

from data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap
from featurebox.featurizers.state.statistics import WeightedAverage, WeightedSum
import pandas as pd

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        ce = CheckElements.from_pymatgen_structures()
        self.data0_checked = ce.check(self.data)[:10]

    def test_WeightedAverage(self):
        data_map = AtomTableMap(search_tp="name", n_jobs=1)
        wa = WeightedAverage(data_map, n_jobs=1)
        x3 = [{"H": 2, }, {"Pd": 1}]
        x4 = wa.fit_transform(x3)
        print(x4)

    def test_WeightedSum(self):
        data_map = AtomTableMap(search_tp="number", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x3 = [[2, 1],[2,3,4]]
        x4 = wa.fit_transform(x3)
        print(x4)

    def test_WeightedSum2(self):
        data_map = AtomJsonMap(search_tp="name", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x3 = [{"H": 2, "Pd": 1}, {"He": 1, "Al": 4}]
        x4 = wa.fit_transform(x3)
        print(x4)

    def test_WeightedSum_stru(self):
        data_map = AtomTableMap(search_tp="number", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x4 = wa.fit_transform(self.data0_checked)
        print(x4)

    def test_WeightedSum2_stru(self):
        data_map = AtomJsonMap(search_tp="name", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x4 = wa.fit_transform(self.data0_checked)
        print(x4)


if __name__ == '__main__':
    unittest.main()

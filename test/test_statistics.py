import unittest

from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap
from featurebox.featurizers.state.statistics import WeightedAverage, WeightedSum


class MyTestCase(unittest.TestCase):

    def test_WeightedAverage(self):
        data_map = AtomTableMap(search_tp="name", n_jobs=1)
        wa = WeightedAverage(data_map, n_jobs=1)
        x3 = [{"H": 2, }, {"Pd": 1}]
        x4 = wa.fit_transform(x3)
        print(x4)

    def test_WeightedSum(self):
        data_map = AtomTableMap(search_tp="name", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x3 = [{"H": 2, "Pd": 1}]
        x4 = wa.fit_transform(x3)
        print(x4)

    def test_WeightedSum2(self):
        data_map = AtomJsonMap(search_tp="name", n_jobs=1)
        wa = WeightedSum(data_map, n_jobs=1)
        x3 = [{"H": 2, "Pd": 1}, {"He": 1, "Al": 4}]
        x4 = wa.fit_transform(x3)
        print(x4)


if __name__ == '__main__':
    unittest.main()

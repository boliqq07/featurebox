import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.envir.environment import BaseNNGet


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = ce.check(self.data)[:3]
        self.data0_checked = ce.check(self.data)[:10]

    def test_get(self):
        bag = BaseNNGet(nn_strategy="VoronoiNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get2(self):
        bag = BaseNNGet(nn_strategy="UserVoronoiNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    # def test_get3(self):
    #     bag = BaseNNGet(nn_strategy="JmolNN")
    #     for i in self.data0_3:
    #         resultt = bag.convert(i)
    #         print(resultt)

    def test_get4(self):
        bag = BaseNNGet(nn_strategy="MinimumDistanceNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get19(self):
        bag = BaseNNGet(cutoff=5.0)
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    # def test_get5(self):
    #     bag = BaseNNGet(nn_strategy="OpenBabelNN")
    #     for i in self.data0_checked:
    #         resultt = bag.convert(i)
    #         print(resultt)

    # def test_get6(self):
    #     bag = BaseNNGet(nn_strategy="CovalentBondNN")
    #     for i in self.data0_3:
    #         resultt = bag.convert(i)
    #         print(resultt)

    # def test_get8(self):
    #     bag = BaseNNGet(nn_strategy="MinimumOKeeffeNN")
    #     for i in self.data0_3:
    #         resultt = bag.convert(i)
    #         print(resultt)

    def test_get9(self):
        bag = BaseNNGet(nn_strategy="BrunnerNN_reciprocal")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get91(self):
        bag = BaseNNGet(nn_strategy="BrunnerNN_real")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get10(self):
        bag = BaseNNGet(nn_strategy="EconNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get11(self):
        bag = BaseNNGet(nn_strategy="CrystalNN")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    # def test_get13(self):
    #     bag = BaseNNGet(nn_strategy="Critic2NN")
    #     for i in self.data0_3:
    #         resultt = bag.convert(i)
    #         print(resultt)


if __name__ == '__main__':
    unittest.main()

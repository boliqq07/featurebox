import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.envir.environment import GEONNGet
from featurebox.featurizers.envir.local_env import UserVoronoiNN
from test.structure_data.get_dataset import data01


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_checked = ce._check(self.data)[:10]

    def test_get(self):
        bag = GEONNGet(UserVoronoiNN)
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get2(self):
        bag = GEONNGet(nn_strategy="wACSF")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get3(self):
        bag = GEONNGet(nn_strategy="SO4_Bispectrum")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get4(self):
        bag = GEONNGet(nn_strategy="SO3")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get5(self):
        bag = GEONNGet(nn_strategy="EAMD")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get6(self):
        bag = GEONNGet(nn_strategy="BehlerParrinello")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get7(self):
        bag = GEONNGet(nn_strategy="EAD")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get8(self):
        bag = GEONNGet(nn_strategy="SOAP")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)


if __name__ == '__main__':
    unittest.main()

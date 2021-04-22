import unittest

from featurebox.data.check_data import CheckElements
import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.envir.environment import BaseNNGet, BaseDesGet
from featurebox.featurizers.envir.local_env import UserVoronoiNN


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_checked = ce.check(self.data)[:10]

    def test_get(self):
        bag = BaseNNGet(UserVoronoiNN)
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get2(self):
        bag = BaseDesGet(nn_strategy="wACSF")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get3(self):
        bag = BaseDesGet(nn_strategy="SO4_Bispectrum")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get4(self):
        bag = BaseDesGet(nn_strategy="SO3")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get5(self):
        bag = BaseDesGet(nn_strategy="EAMD")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get6(self):
        bag = BaseDesGet(nn_strategy="BehlerParrinello")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get7(self):
        bag = BaseDesGet(nn_strategy="EAD")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)

    def test_get8(self):
        bag = BaseDesGet(nn_strategy="SOAP")
        for i in self.data0_3:
            resultt = bag.convert(i)
            print(resultt)


if __name__ == '__main__':
    unittest.main()

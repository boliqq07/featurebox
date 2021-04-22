import unittest

import numpy as np
import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap, AtomPymatgenPropMap
from featurebox.featurizers.base_transform import ConverterCat
from featurebox.featurizers.envir.environment import BaseDesGet


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        ce = CheckElements.from_pymatgen_structures()
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = ce.check(self.data)[:10]
        self.data0_checked = ce.check(self.data)[:10]

    def test_get(self):
        bag = BaseDesGet()
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
        for i in self.data0_checked:
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

    def test_cat(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomTableMap(search_tp="name"))
        s = [{"H": 2, }, {"Po": 1}]
        b = tmps.convert(s)
        assert b.shape[1] > 20

    def test_cat2(self):
        tmps = ConverterCat(AtomTableMap(search_tp="number"), AtomTableMap(search_tp="number"))
        s = [1, 2]
        b = tmps.convert(s)
        assert b.shape[1] > 20

    def test_cat3(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomJsonMap())
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] > 20

    def test_cat4(self):
        tmps = ConverterCat(AtomTableMap(search_tp="name"), AtomJsonMap("ie.json"))
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] >= 20

    def test_cat5(self):
        tmps = ConverterCat(AtomJsonMap(), AtomJsonMap("ie.json"))
        s = [{"H": 2, }, {"Pd": 1}]
        b = tmps.convert(s)
        print(b.shape)
        assert b.shape[1] >= 17

    def test_equal(self):
        tmps = AtomJsonMap("ie.json")
        s = [{"H": 1, }, {"Pd": 1}]
        b = tmps.convert(s)
        tmps = AtomJsonMap("ie.json", search_tp="number")
        s = [1, 46]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))

    def test_equal2(self):
        tmps = AtomTableMap(search_tp="name")
        s = [{"As": 1, }, {"U": 1}]
        b = tmps.convert(s)
        tmps = AtomTableMap(search_tp="number")
        s = [33, 92]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))

    def test_equal3(self):
        tmps = AtomPymatgenPropMap("X", search_tp="name")
        s = [{"As": 1, }, {"U": 1}]
        b = tmps.convert(s)
        tmps = AtomPymatgenPropMap("X", search_tp="number")
        s = [33, 92]
        c = tmps.convert(s)
        self.assertTrue(np.all(np.equal(b, c)))


if __name__ == '__main__':
    unittest.main()

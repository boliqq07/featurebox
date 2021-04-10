import unittest
import pandas as pd
import numpy as np

from featurebox.featurizers.base_graph import CrystalGraph, CrystalGraphWithBondTypes, CrystalGraphDisordered


class TestCrystalGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_CrystalGraph_convert_call(self):
        sg1 = CrystalGraph()
        s11 = sg1(self.data0)
        s12 = sg1(self.data0, state_attributes=np.array([2,3.0]))

        self.assertTrue(isinstance(s12,dict))
        self.assertEqual(list(s12.keys()), ['atom', 'bond', 'state', 'atom_nbr_idx'])
        for i in s12.values():
            print(type(i))
            self.assertTrue(isinstance(i, (np.ndarray,list)))

    def test_CrystalGraph_as_dict(self):
        sg1 = CrystalGraph()
        dict1 = sg1.as_dict()
        sg2 = CrystalGraph.from_dict(dict1)
        s12 = sg2(self.data0)


class TestCrystalGraphWithBondTypes(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_CrystalGraph_convert_call(self):
        sg1 = CrystalGraphWithBondTypes()
        s11 = sg1(self.data0)
        s12 = sg1(self.data0, state_attributes=np.array([2,3.0]))

        self.assertTrue(isinstance(s12,dict))
        self.assertEqual(list(s12.keys()), ['atom', 'bond', 'state', 'atom_nbr_idx'])
        for i in s12.values():
            print(type(i))
            self.assertTrue(isinstance(i, (np.ndarray,list)))

    def test_CrystalGraph_as_dict(self):
        sg1 = CrystalGraphWithBondTypes()
        dict1 = sg1.as_dict()
        sg2 = CrystalGraphWithBondTypes.from_dict(dict1)
        s12 = sg2(self.data0)


class TestCrystalGraphDisordered(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_CrystalGraph_convert_call(self):
        sg1 = CrystalGraphDisordered()
        s11 = sg1(self.data0)
        s12 = sg1(self.data0, state_attributes=np.array([2,3.0]))

        self.assertTrue(isinstance(s12,dict))
        self.assertEqual(list(s12.keys()), ['atom', 'bond', 'state',  'atom_nbr_idx'])
        for i in s12.values():
            print(type(i))
            self.assertTrue(isinstance(i, (np.ndarray,list)))

    def test_CrystalGraph_as_dict(self):
        sg1 = CrystalGraphDisordered()
        dict1 = sg1.as_dict()
        sg2 = CrystalGraphDisordered.from_dict(dict1)
        s12 = sg2(self.data0)


if __name__ == '__main__':
    unittest.main()

import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.base_graph import CrystalGraph
from featurizers.base_graph_geo import StructureGraphGEO


class TestGraph3(unittest.TestCase):
    def setUp(self) -> None:

        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_3 = self.data[:3]
        ce = CheckElements.from_pymatgen_structures()

        self.data0_checked = ce.check(self.data)[:10]

    def test_CrystalGraph(self):
        for i in self.data0_checked:
            sg1 = StructureGraphGEO(nn_strategy="find_points_in_spheres",
                 bond_generator=None,
                 atom_converter = None,
                 bond_converter = None,
                 state_converter = None,
                 return_bonds = "all",
                 cutoff = 5.0,)
            s12 = sg1(i)
            # print(s12)
            print(s12["bond"].shape[-2])
            print(s12["bond"].shape[-1])

    def test_CrystalGraph2(self):
        for i in self.data0_checked:
            sg1 = StructureGraphGEO(nn_strategy="find_xyz_in_spheres",
                 bond_generator=None,
                 atom_converter = None,
                 bond_converter = None,
                 state_converter = None,
                 return_bonds = "all",
                 cutoff = 5.0,)
            s12 = sg1(i)
            # print(s12)
            print("next")
            for key in s12.keys():
                print(s12[key].shape)

    def test_CrystalGraph3(self):
        for i in self.data0_checked:
            sg1 = StructureGraphGEO(nn_strategy="CrystalNN",
                 bond_generator=None,
                 atom_converter = None,
                 bond_converter = None,
                 state_converter = None,
                 return_bonds = "all",
                 cutoff = 5.0,)
            s12 = sg1(i)
            # print(s12)
            print("next")
            for key in s12.keys():
                print(s12[key].shape)

if __name__ == '__main__':
    unittest.main()

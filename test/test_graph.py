import unittest

import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.base_graph import CrystalGraph
from featurizers.bond.smooth import Smooth


class TestGraph2(unittest.TestCase):
    def setUp(self) -> None:

        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]
        self.data0_3 = self.data[:3]
        ce = CheckElements.from_pymatgen_structures()

        self.data0_checked = ce.check(self.data)[:10]

    def test_CrystalGraph(self):
        for i in self.data0_checked:
            sg1 = CrystalGraph(nn_strategy="SOAP", bond_generator="BaseDesGet", cutoff=None)
            s12 = sg1(i)
            # print(s12)
            print(s12["bond"].shape[-2])
            print(s12["bond"].shape[-1])

    def test_CrystalGraph2(self):
        for i in self.data0_checked:
            sg1 = CrystalGraph(nn_strategy="EAMD", bond_generator="BaseDesGet", cutoff=None)
            s12 = sg1(i)
            # print(s12)
            print(s12["bond"].shape[-2])
            print(s12["bond"].shape[-1])

    def test_CrystalGraphsmooth(self):
        for i in self.data0_checked:
            sg1 = CrystalGraph(nn_strategy="find_xyz_in_spheres",
                               return_bonds="bonds",
                               cutoff=5.0,
                               bond_converter=Smooth(r_c=5.0, r_cs=3.0)
                               )
            s12 = sg1(i)
            # print(s12)
            print(s12["bond"].shape[-2])
            print(s12["bond"].shape[-1])


if __name__ == '__main__':
    unittest.main()

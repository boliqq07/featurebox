import unittest

import pandas as pd

from featurebox.featurizers.base_graph import CrystalGraph


# import pandas as pd
# preprocessing = pd.read_pickle("data_structure.pkl_pd")
#
# sg1 = CrystalGraph()
# s12 = sg1(preprocessing[0])
#
# sg2 = CrystalGraphDisordered()
# s22 = sg2(preprocessing[0],[1.0,2.0])
#
# sg3 = CrystalGraphWithBondTypes()
# s33 = sg3(preprocessing[0],[1.0, 2.0])
#
#
# sg1 = CrystalGraph()
# s10 = [sg1(preprocessing[i]) for i in range(10)]
#
# # GraphSingleGenerator()
# datax=s10


class TestGraph2(unittest.TestCase):
    def setUp(self) -> None:

        self.data = pd.read_pickle("data_structure.pkl_pd")
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_CrystalGraph(self):
        sg1 = CrystalGraph()
        s12 = sg1(self.data0)
        print(s12)


if __name__ == '__main__':
    unittest.main()
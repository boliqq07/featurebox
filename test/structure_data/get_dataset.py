# -*- coding: utf-8 -*-

# @Time    : 2021/8/5 22:55
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pandas as pd
from pymatgen.core import Structure
data01 = pd.read_pickle("/home/iap13/wcx/featurebox/test/structure_data/data_structure.pkl_pd")
data02 = pd.read_pickle("/home/iap13/wcx/featurebox/test/structure_data/data_structure2.pkl_pd")
data03 = Structure.from_file("/home/iap13/wcx/featurebox/test/structure_data/674718.cif")
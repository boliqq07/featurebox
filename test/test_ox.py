import unittest

import numpy as np
import pandas as pd

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap, AtomPymatgenPropMap, BinaryMap
from featurebox.featurizers.base_transform import ConverterCat
from featurebox.featurizers.envir.environment import BaseDesGet
from pymatgen.core import Structure

# class TestGraph(unittest.TestCase):
#     def setUp(self) -> None:
#         self.st = Structure.from_file("674718.cif")
#         self.st2 = [Structure.from_file("674718.cif"),Structure.from_file("674718.cif")]
#
#     def test_convert(self):
#         atm = AtomTableMap()
#         atm.convert_structure(self.st)
#         atm.convert_structure(self.st)

    # def test_convert(self):
    #     pass


st = Structure.from_file("674718.cif")
st2 = [Structure.from_file("674718.cif"),Structure.from_file("674718.cif")]

atm = AtomJsonMap()
atm.convert(st)
# atm.convert_structure(st)
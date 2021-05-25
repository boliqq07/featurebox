import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap, AtomPymatgenPropMap, BinaryMap
from featurebox.featurizers.base_transform import ConverterCat
from featurebox.featurizers.envir.environment import BaseDesGet


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.st = Structure.from_file("674718.cif")
        self.st2 = [Structure.from_file("674718.cif"),Structure.from_file("674718.cif")]

    def test_convert(self):
        atm = AtomTableMap()
        atm.convert_structure(self.st)

    def test_convert2(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.convert(self.st)

    def test_tra(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.transform(self.st2)



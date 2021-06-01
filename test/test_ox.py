import unittest

from pymatgen.core import Structure

from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.st = Structure.from_file("674718.cif")
        self.st2 = [Structure.from_file("674718.cif"), Structure.from_file("674718.cif")]

    def test_convert(self):
        atm = AtomTableMap()
        a = atm.convert(self.st)
        print(a)

    def test_convert2(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.convert(self.st)

    def test_tra(self):
        atm = AtomJsonMap(search_tp="name")
        re = atm.transform(self.st2)

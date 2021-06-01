import unittest

# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:20:52 2021

@author: GL
"""

import os

from pymatgen.core import Structure

from featurebox.data.check_data import CheckElements
from featurebox.featurizers.atom.mapper import AtomTableMap
from featurebox.featurizers.base_graph import CrystalGraph


class MyTestCase(unittest.TestCase):

    def test_ion(self):
        os.chdir('./test_gl')
        PATH = os.getcwd()
        print(PATH)
        from mgetool.imports import BatchFile

        bf = BatchFile(os.path.join(PATH, "data"), suffix='cif')
        f = bf.merge()
        os.chdir(PATH)
        data = [Structure.from_file(i) for i in f[:10]]
        ce = CheckElements.from_pymatgen_structures()
        checked_data = ce.check(data)

        tmps = AtomTableMap(search_tp="name")
        gt = CrystalGraph(n_jobs=2, atom_converter=tmps)
        in_data = gt.transform(checked_data, state_attributes=None)


if __name__ == '__main__':
    unittest.main()

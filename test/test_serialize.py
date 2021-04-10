
import unittest

from featurebox.featurizers.local_env import *
from pymatgen.analysis.local_env import VoronoiNN
from monty.json import MSONable

class AD(MSONable):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class TestGraph(unittest.TestCase):
    def test_ser(self):

        ad = AD(a=5, b=NearNeighbors_D())
        #
        # ad = serialize(VoronoiNN(cutoff=5.0))
        afd = ad.as_dict()
        afj = ad.to_json()
        ad2 = ad.from_dict(afd)

    def test_ser2(self):
        a = get_nn_strategy(VoronoiNN)
        print(a.__class__)
if __name__ == '__main__':
    unittest.main()


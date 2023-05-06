import unittest

from featurebox.featurizers.batch_feature import BatchFeature
from test.structure_data.get_dataset import data01


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data01
        self.data0 = self.data[0]
        self.data0_3 = self.data[:3]

    def test_something2(self):
        sppm = BatchFeature(data_type="compositions", return_type="df")
        data = sppm.fit_transform([i.composition.to_reduced_dict for i in self.data0_3])
        print(data)

    def test_something3(self):
        sppm = BatchFeature(data_type="elements")
        aa = []
        aas = [[i.species.as_dict() for i in structure.sites] for structure in self.data0_3]
        [aa.extend(i) for i in aas]
        data = sppm.fit_transform([aa])
        data = sppm.fit_transform([aa])

    def test_something4(self):
        structures_list = self.data0_3
        sppm = BatchFeature(data_type="structures", return_type="df")
        data = sppm.fit_transform(structures_list)


if __name__ == '__main__':
    unittest.main()

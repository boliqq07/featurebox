# -*- coding: utf-8 -*-

# @Time    : 2019/11/1 15:57
# @Email   : 986798607@qq.ele_ratio
# @Software: PyCharm
# @License: BSD 3-Clause
import numpy as np

from featurebox.featurizers.extrastats import PropertyStats


class UnionFeature:
    """
    transform method should input0 comp_index rather than entries
    """

    def __init__(self, comp, couple_data, couple=2, stats=("mean",)):
        self.couple = couple
        self.comp = comp
        self.elem_data = couple_data
        self.stats = stats
        # Initialize stats computer

    def fit_transform(self, comp_number=0):
        """
        Get elemental property attributes

        Args:
            comp: Pymatgen composition object

        Returns:
            all_attributes: Specified property statistics of features
            :param comp_number:
        """
        comp = self.comp[comp_number]
        elem_data = self.elem_data[comp_number].values

        # Get the element names and fractions
        elements, fractions = zip(*comp.element_composition.items())
        elem_data = np.reshape(elem_data, (self.couple, -1), order="F")
        all_attributes_all = []
        for stat in self.stats:
            all_attributes = [PropertyStats.calc_stat(elem_data_i, stat, fractions) for elem_data_i in elem_data.T]
            all_attributes_all.extend(all_attributes)
        return np.array(all_attributes_all)

    def feature_labels(self):
        """
        Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        name = np.array(self.elem_data.columns.values)[::self.couple]
        name = [i.split("_")[0] + "%s" % j for i in name for j in self.stats]
        return name

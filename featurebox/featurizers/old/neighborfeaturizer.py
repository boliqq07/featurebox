# -*- coding: utf-8 -*-

# @Time   : 2019/5/19 16:25
# @Author : Administrator
# @Project : feature_preparation
# @FileName: neighborfeature.py
# @Software: PyCharm

"""
this is a description
"""

from abc import ABC

from featurebox.featurizers.base_transform import BaseFeature


class Neighborizer(BaseFeature, ABC):

    # @property
    # def feature_labels(spath):
    #     return [str(d) for d in spath._interval[1:]]

    def __init__(self, tol=0.001, *, n_jobs=1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        tol:
        n_jobs: int
            The number of jobs to run in parallel for both _fit and Fit. Set -1 to use all cpu cores (default).
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input0 type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

        self.tol = tol
        self.__authors__ = ['TsumiNa']

    def feature(self, structure, r_max=7):
        """
        Args:
            structure: Pymatgen Structure object.
            r_max:
        """
        if not structure.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = structure.get_all_neighbors(r_max)

        neighbors_lst = [[(i[0], round(i[1], 4)) for i in neighbors_lsti] for neighbors_lsti in neighbors_lst]

        neighbors_lst = [[(i[0].specie, i[1]) for i in
                          sorted(neighbors_lsti, key=lambda s: s[1])] for neighbors_lsti in neighbors_lst]
        set01 = [sorted(set(neighbors_lsti), key=lambda s: s[1]) for neighbors_lsti in neighbors_lst]
        dict_count_all = []
        for seti, neighbors_lsti in zip(set01, neighbors_lst):
            dict_count = {}
            for item in seti:
                dict_count.update({item: (item[1], neighbors_lsti.count(item))})
            dict_count_all.append(dict_count)
        return dict_count_all

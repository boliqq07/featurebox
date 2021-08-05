#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/27 16:57
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Calculate the the correction of columns.
"""
import copy
from typing import List

import numpy as np
from mgetool.tool import name_to_name
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from featurebox.selection.mutibase import MutiBase


class Corr(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MutiBase):
    """
    Calculate correlation. (Where the result are changed with random state.)

    Examples
    ---------
    >>> from sklearn.datasets import load_boston
    >>> from featurebox import Corr
    >>> x, y = load_boston(return_X_y=True)
    >>> co = Corr(threshold=0.5)
    >>> nx = co.fit_transform(x)

    **1. Used for get group exceeding the threshold**

    Examples
    ---------

    >>> from sklearn.datasets import load_boston
    >>> from featurebox import Corr
    >>> x, y = load_boston(return_X_y=True)
    >>> co = Corr(threshold=0.7)
    >>> groups = co.count_cof(np.corrcoef(x[:,:7], rowvar=False))
    >>> groups[1]
    [[0], [1], [2, 4], [3], [2, 4, 6], [5], [4, 6]]
    >>> groups[0]
    [[1.0], [1.0], [1.0, 0.764], [1.0], [0.764, 1.0, 0.731], [1.0], [0.731, 1.0]]

    Where the (2,4), (2,4,6), (4,6) are with correlation more than 0.7.

    **2. Used for filter automatically by machine**

    Examples
    -----------
    >>> from sklearn.datasets import load_boston
    >>> from featurebox import Corr
    >>> x, y = load_boston(return_X_y=True)
    >>> co = Corr(threshold=0.7)
    >>> co.fit(x)
    Corr(threshold=0.7)
    >>> group = co.count_cof()
    >>> group[1]
    [[0], [1], [2, 4, 7, 9], [3], [2, 4, 6, 7], [5], [4, 6, 7], [2, 4, 6, 7], [8, 9], [2, 8, 9], [10], [11], [12]]
    >>> co.remove_coef(group[1]) # Filter automatically by machine.
    [0, 1, 2, 3, 5, 6, 8, 10, 11, 12]

    Where the remove_coef are changed with random state.

    **3. Used for binding correlation**

    Examples
    -----------
    >>> from sklearn.datasets import load_boston
    >>> from featurebox import Corr
    >>> x, y = load_boston(return_X_y=True)
    >>> co = Corr(threshold=0.7,muti_index=[0,8],muti_grade=2)
    >>> # in range [0,8], the features are binding in to 2 sized: [[0,1],[2,3],[4,5],[6,7]]
    >>> co.fit(x)
    Corr(muti_index=[0, 8], threshold=0.7)
    >>> group = co.count_cof()
    >>> group[1]
    [[0], [1], [2], [3], [4, 5], [4, 5], [6], [7], [8]]
    >>> co.remove_coef(group[1]) # Filter automatically by machine.
    [0, 1, 2, 3, 4, 6, 7, 8]

    Where 4 is filtered , Corresponding to the initial feature 8.

    [0,1] -> 0; [2,3] -> 1; [4,5]->2; [6,7]->3, 8->4; 9->5; 10->6; 11->7; 12->8; 13->9;
    """

    def __init__(self, threshold: float = 0.85, muti_grade: int = 2, muti_index: List = None, must_index: List = None,
                 random_state: int = 0):
        """

        Parameters
        ----------
        threshold:float
            ranking threshold.
        muti_grade:
            binding_group size, calculate the correction between binding.
        muti_index:list
            the range of muti_grade:[min,max).
        must_index:list
            the columns force to index.
        random_state:int
            int
        """

        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        self.threshold = threshold
        self.cov = None
        self.cov_shrink = None
        self.shrink_list = []
        self.random_state = random_state

    def fit(self, data, pre_cal=None, method="mean"):
        if pre_cal is None:

            cov = np.corrcoef(data, rowvar=False, )

        elif isinstance(pre_cal, np.ndarray) and pre_cal.shape[0] == data.shape[1]:
            cov = pre_cal
        else:
            raise TypeError("pre_cal is None or coef of data_cluster with shape(data_cluster[0],data_cluster[0])")
        cov = np.nan_to_num(cov - 1) + 1
        self.cov = cov
        self.data = data
        self.shrink_list = list(range(self.cov.shape[0]))
        self._shrink_coef(method=method)
        self.filter()
        return self

    def _shrink_coef(self, method="mean" or "max"):

        if self.check_muti:
            self.shrink_list = list(range(self.cov.shape[0]))
            self.shrink_list = list(self.feature_fold(self.shrink_list))

            cov = self.cov
            single = tuple([i for i in self.shrink_list if i not in self.check_muti])
            muti = tuple([i for i in self.shrink_list if i in self.check_muti])

            cov_muti_all = []
            le = self.muti_grade
            while le:
                index = []
                index.extend(single)
                index.extend([i + le - 1 for i in muti])
                index.sort()
                cov_muti_all.append(cov[index][:, index])
                le -= 1
            cov_muti_all = np.array(cov_muti_all)
            if method == "mean":
                cov_new = np.mean(cov_muti_all, axis=0)
            else:
                cov_new = np.max(cov_muti_all, axis=0)
            self.cov_shrink = cov_new
            return self.cov_shrink
        else:
            self.cov_shrink = self.cov

    def count_cof(self, cof=None):
        """Check cof and count the number."""
        if cof is None:
            cof = self.cov_shrink
        if cof is None:
            raise NotImplemented("imported cof is None")

        list_count = []
        list_coef = []
        g = np.where(abs(cof) >= self.threshold)
        for i in range(cof.shape[0]):
            e = np.where(g[0] == i)
            com = list(g[1][e])
            # ele_ratio.remove(i)
            list_count.append(com)
            list_coef.append([round(cof[i, j], 3) for j in com])
        self.list_coef = list_coef
        self.list_count = list_count
        return list_coef, list_count

    def remove_coef(self, cof_list_all):
        """Delete the index of feature with repeat coef."""
        ran = check_random_state(self.random_state)
        reserve = []
        for i in cof_list_all:
            if not cof_list_all:
                reserve.append(i)

        for cof_list in cof_list_all:
            if not cof_list:
                pass
            else:
                if reserve:
                    candi = []
                    for j in cof_list:

                        con = any([[False, True][j in cof_list_all[k]] for k in reserve])
                        if not con:
                            candi.append(j)
                    if any(candi):
                        a = ran.choice(candi)
                        reserve.append(a)
                    else:
                        pass
                else:
                    a = ran.choice(cof_list)
                    reserve.append(a)
                cof_list_t = copy.deepcopy(cof_list)
                for dela in cof_list_t:
                    for cof_list2 in cof_list_all:
                        if dela in cof_list2:
                            cof_list2.remove(dela)
        return sorted(list(set(reserve)))

    def filter(self):
        list_coef, list_count = self.count_cof()
        index = self.remove_coef(list_count)
        support_ = [self.shrink_list[i] for i in index]
        support_ = self.feature_unfold(support_)
        self.support_ = np.array([True if i in support_ else False for i in range(self.data.shape[1])])
        return support_

    @staticmethod
    def cov_y(x_, y_):
        cov = np.corrcoef(x_, y_, rowvar=False)
        cov = cov[:, -1][:-1]
        return cov

    def remove_by_y(self, y_):
        corr = self.cov_y(self.data, y_)
        corr = self.feature_fold(corr)
        lcount = self.list_count
        fea_all = []
        score = name_to_name(corr, search=lcount, search_which=0, return_which=(1,), two_layer=True)
        for score_i, list_i in zip(score, lcount):
            indexs = np.argmax(score_i)
            feature_index = list_i[int(indexs)]
            fea_all.append(feature_index)

        fea_all = sorted(list(set(fea_all)))
        return fea_all

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    # def transform_index(self, data):
    #     if isinstance(data, int):
    #         return self.shrink_list.index(data)
    #     elif isinstance(data, (list, tuple)):
    #         return [self.shrink_list.index(i) for i in data]
    #
    # def inverse_transform_index(self, data):
    #     if isinstance(data, int):
    #         return self.shrink_list[data]
    #     elif isinstance(data, (list, tuple)):
    #         return [self.shrink_list[i] for i in data]
    #     else:
    #         pass
    #
    # def transform(self, data):
    #     if isinstance(data, (list, tuple)):
    #         return data[self.shrink_list]
    #     elif isinstance(data, np.ndarray) and data.ndim == 1:
    #         return data[self.shrink_list]
    #     elif isinstance(data, np.ndarray) and data.ndim == 2:
    #         return data[:, self.shrink_list]
    #     else:
    #         pass


if __name__ == "__main__":
    # x, y = load_boston(return_X_y=True)
    # co = Corr(threshold=0.7)
    # c = co.count_cof(np.corrcoef(x, rowvar=False))[1]

    x, y = load_boston(return_X_y=True)
    co = Corr(threshold=0.5, muti_index=[0, 8], muti_grade=2)

    nx = co.fit_transform(x)

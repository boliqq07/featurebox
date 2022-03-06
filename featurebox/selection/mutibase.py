# -*- coding: utf-8 -*-

# @Time   : 2019/5/26 0:47
# @Author : Administrator
# @Project : feature_preparation
# @FileName: mutibase.py
# @Software: PyCharm
from typing import List

import numpy as np


class MutiBase(object):
    """Base method for binding"""

    def __init__(self, muti_grade: int = 2, muti_index: List = None, must_index: List = None):
        """

        Parameters
        ----------
        muti_grade:
            binding_group size, calculate the correction between binding
        muti_index:list
            the range of muti_grade:[min,max)
        must_index:list,None
            the columns force to index
        """
        self.muti_grade = muti_grade
        self.muti_index = muti_index
        self.must_index = must_index

    @property
    def check_muti(self):
        muti_index = self.muti_index
        if muti_index is None:
            return False
        elif isinstance(muti_index, (list, tuple)):
            if len(muti_index) == 2 and isinstance(muti_index[0], int) and isinstance(muti_index[1], int):
                return tuple(range(*muti_index))
        else:
            raise TypeError("muti_index should be None or iterable type with 2 number")

    @property
    def check_must(self):
        must_index = self.must_index
        if must_index is None:
            return False
        elif isinstance(must_index, (list, tuple)):
            if all([isinstance(_, int) for _ in must_index]) and 0 < len(must_index) <= 2:
                if len(must_index) == 1:
                    must_slice = tuple(must_index)
                else:
                    must_slice = tuple(range(*must_index))
                return must_slice
        else:
            raise TypeError("must_index should be None or iterable type with less than 2 number")

    def feature_fold(self, feature):
        muti_grade, muti_index = self.muti_grade, self.muti_index
        if self.check_muti:
            feature = np.sort(feature)
            single = np.array([_ for _ in feature if _ < muti_index[0] or _ >= muti_index[1]])
            com_com = np.array([_ for _ in feature if muti_index[1] > _ >= muti_index[0]])
            com_sin = com_com[::muti_grade]
            return np.sort(np.hstack((single, com_sin))).astype(int)
        else:
            return np.sort(feature).astype(int)

    def feature_unfold(self, feature):
        muti_grade, muti_index = self.muti_grade, self.muti_index
        if self.check_muti:
            single = np.array([_ for _ in feature if _ < muti_index[0] or _ >= muti_index[1]])
            com_sin = np.array([_ for _ in feature if muti_index[1] > _ >= muti_index[0]])
            com_com = list(com_sin)
            while muti_grade - 1:
                com_com.extend(com_sin + (muti_grade - 1))
                muti_grade -= 1
            return np.sort(list(set(np.hstack((single, np.array(com_com)))))).astype(int)
        else:
            return np.sort(np.array(feature)).astype(int)

    def feature_must_fold(self, feature):
        must_index = self.must_index
        if must_index:
            feature = list(feature)
            if len(must_index) == 1 and must_index[0] not in feature:
                feature.append(must_index[0])
            else:
                must_feature = list(range(*must_index))
                must_feature = self.feature_fold(must_feature)
                feature.extend([j for j in must_feature if j not in feature])
            return np.sort(feature).astype(int)
        else:
            return np.sort(feature).astype(int)

    def feature_must_unfold(self, feature):
        must_index = self.must_index
        if must_index:
            feature = list(feature)
            if len(must_index) == 1 and must_index[0] not in feature:
                feature.append(must_index[0])
            else:
                must_feature = list(range(*must_index))
                feature.extend([j for j in must_feature if j not in feature])
            return np.sort(feature).astype(int)
        else:
            return np.sort(feature).astype(int)

    def inverse_transform_index(self, index):
        """inverse the selected index to origin index by support."""
        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            index = np.where(index)[0]
        index = np.array(list(index))

        return np.where(self.support_)[0][index]

    def transform_index(self, index):
        """Get support index."""
        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            index = np.where(index)[0]
        return np.array([i for i in index if self.support_[i]])

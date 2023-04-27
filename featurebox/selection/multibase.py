# -*- coding: utf-8 -*-

# @Time   : 2019/5/26 0:47
# @Author : Administrator
# @Project : feature_preparation
# @FileName: multibase.py
# @Software: PyCharm
import itertools
from typing import List, Optional, Tuple, Sequence, Any, Union

import numpy as np


class MultiBase(object):
    """Base method for binding"""

    def __init__(self, multi_grade: int = 2, multi_index: Optional[Union[List, Tuple]] = None,
                 must_index: Optional[Union[List, Tuple]] = None):
        """

        Parameters
        ----------
        multi_grade: int
            binding_group size, calculate the correction between binding
        multi_index: list,tuple,None
            the range of multi_grade:[min,max)
        must_index: list,tuple,None
            the columns force to index
        """
        self.multi_grade = multi_grade
        self.multi_index = tuple(multi_index) if multi_index is not None else multi_index
        self.must_index = tuple(must_index) if must_index is not None else must_index

    @property
    def check_multi(self):
        multi_index = self.multi_index
        if multi_index is None:
            return False
        elif isinstance(multi_index, (list, tuple)):
            if len(multi_index) == 2 and isinstance(multi_index[0], int) and isinstance(multi_index[1], int):
                return True
            else:
                raise TypeError("multi_index should be None or iterable type with 2 number.")
        else:
            raise TypeError("multi_index should be None or iterable type with 2 number.")

    @property
    def check_must(self):
        must_index = self.must_index
        if must_index is None:
            return False
        elif isinstance(must_index, (list, tuple)):
            return True
        else:
            raise TypeError("must_index should be None or iterable type with less than 1 number")

    @property
    def must_fold_add(self):
        if self.check_must:
            must_index = self.must_index
            if self.check_multi:
                def ff(mi):
                    data = mi - self.multi_index[0]
                    data = (data // self.multi_grade) * self.multi_grade + self.multi_index[0]
                    return data

                com_mark = [self.multi_index[0] <= mi < self.multi_index[1] for mi in must_index]
                must_feature = [ff(mi) if com_mark is True else mi for com_mark_i, mi in zip(com_mark, must_index)]
                return must_feature
            else:
                return list(must_index)
        else:
            return []

    @property
    def must_unfold_add(self):
        if self.check_must:
            must_index = self.must_index
            if self.check_multi:
                def ff2(mi):
                    data = mi - self.multi_index[0]
                    data = (data // self.multi_grade) * self.multi_grade + self.multi_index[0]
                    return list(range(data, data + self.multi_grade))

                com_mark = [self.multi_index[0] <= mi < self.multi_index[1] for mi in must_index]
                must_feature = [ff2(mi) if com_mark is True else [mi, ] for com_mark_i, mi in zip(com_mark, must_index)]
                must_feature = list(itertools.chain(*must_feature))
                return must_feature
            else:
                return list(must_index)
        else:
            return []

    def _feature_fold(self, feature, raw=False):
        multi_grade, multi_index = self.multi_grade, self.multi_index
        feature = list(feature)
        if self.check_multi:
            feature.sort()
            single = [_ for _ in feature if _ < multi_index[0] or _ >= multi_index[1]]
            com_com = [_ for _ in feature if multi_index[1] > _ >= multi_index[0]]
            com_sin = com_com[::multi_grade]
            single.extend(com_sin)
            res = single
        else:
            res = feature

        if not raw:
            res.sort()

        return res

    def feature_fold(self, feature):
        fea2 = list(self._feature_fold(feature))
        add = self.must_fold_add
        fea2.extend(add)
        fea2 = list(set(fea2))
        fea2.sort()
        return np.array(fea2)

    def _feature_unfold(self, feature, raw=False):
        multi_grade, multi_index = self.multi_grade, self.multi_index
        feature = list(feature)
        if self.check_multi:
            single = [_ for _ in feature if _ < multi_index[0] or _ >= multi_index[1]]
            com_sin = [_ for _ in feature if multi_index[1] > _ >= multi_index[0]]
            com_com = list(com_sin)
            while multi_grade - 1:
                multi_grade -= 1
                com_com.extend([i + multi_grade for i in com_sin])
            single.extend(com_com)
            res = list(set(single))
        else:
            res = feature

        if not raw:
            res.sort()

        return res

    def feature_unfold(self, feature):
        fea2 = list(self._feature_unfold(feature))
        add = self.must_unfold_add
        fea2.extend(add)
        fea2 = list(set(fea2))
        fea2.sort()
        return np.array(fea2)

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

    def transform(self, data: Any):
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return data[self.support_]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            return data[:, self.support_]
        elif isinstance(data, Sequence):
            return data[self.support_]
        else:
            raise ValueError

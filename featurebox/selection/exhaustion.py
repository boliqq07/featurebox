# -*- coding: utf-8 -*-

# @Time   : 2019/5/25 17:28
# @Author : Administrator
# @Project : feature_preparation
# @FileName: exhaustion.py
# @Software: PyCharm

import warnings
from functools import partial
from itertools import combinations
from typing import Tuple, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, check_X_y

from featurebox.selection.mutibase import MutiBase
from mgetool.tool import parallelize

warnings.filterwarnings("ignore")


class Exhaustion(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MutiBase):
    """
    Exhaustion features combination.

    The attribute ``estimator_`` is the model with the best feature rather than all feature combination.

    Examples
    ----------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.svm import SVR
    >>> X,y = load_boston(return_X_y=True)
    >>> svr= SVR()
    >>> bf = Exhaustion(svr,n_select=(2,),refit=True)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False, False, False, False, False, False, False, False, False,
           False,  True, False,  True])

    Examples
    ----------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.svm import SVR
    >>> X,y = load_boston(return_X_y=True)
    >>> svr= SVR()
    >>> from sklearn import model_selection
    >>> gd = model_selection.GridSearchCV(svr, param_grid=[{"C": [1, 10]}], n_jobs=1, cv=5)
    >>> bf = Exhaustion(gd,n_select=(2,),refit=True)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    ...
    """

    def __init__(self, estimator: BaseEstimator, n_select: Tuple = (2, 3, 4), muti_grade: int = None,
                 muti_index: List = None, must_index: List = None, n_jobs: int = 1,
                 refit: bool = False, cv: int = 5):
        """

        Parameters
        ----------
        estimator:
            sklearn model or GridSearchCV
        n_select:tuple
            the n_select list,default,n_select=(3, 4)
        muti_grade:list
            binding_group size, calculate the correction between binding
        muti_index:list
            the range of muti_grade:[min,max)
        must_index:list
            the columns force to index
        n_jobs:int
            n_jobs
        refit:bool
            refit or not, if refit the model would used all data.
        cv:bool
            if estimator is sklearn model, used cv, else pass
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        if hasattr(estimator, "max_features") and refit:
            print("For estimator with 'max_features' attribute, the 'max_features' would changed with "
                  "each sub-data. that is, The 'refit estimator' which with fixed 'max_features' could be with different performance.\n"
                  "Please change and define 'max_features' (with other parameters fixed) by manual testing after Exhaustion!!!!\n"
                  "Please change and define 'max_features' (with other parameters fixed) by manual testing after Exhaustion!!!!",
                  )
        self.estimator = estimator
        self.score_ = []
        self.n_jobs = n_jobs
        self.n_select = [n_select, ] if isinstance(n_select, int) else n_select
        self.refit = refit
        self.cv = cv

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the baf model and then the underlying estimator on the selected feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            The training input0 samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, x, y):

        estimator = clone(self.estimator)

        def score_pri(slices, x0, y0):
            slices = list(slices)
            if len(slices) < 1:
                score0 = - np.inf
            else:
                slices = self.feature_unfold(slices)
                data_x0 = x0[:, slices]

                if hasattr(self.estimator, "max_features"):
                    if self.estimator.max_features > data_x0.shape[1]:
                        estimator_ = clone(self.estimator)
                        estimator_.max_features = data_x0.shape[1]

                    else:
                        estimator_ = estimator
                else:
                    estimator_ = estimator

                if hasattr(estimator_, "best_score_"):
                    estimator_.fit(data_x0, y0)
                    score0 = np.mean(estimator_.best_score_)  # score_test
                elif self.cv == 0:
                    estimator_.fit(data_x0, y0)
                    score0 = estimator_.score(data_x0, y0)
                else:
                    score0 = cross_val_score(estimator_, data_x0, y0, cv=self.cv)
                    score0 = np.mean(score0)
                # print(slices, score0)
            return score0

        score = partial(score_pri, x0=x, y0=y)

        self.score_ = []
        x, y = check_X_y(x, y, "csc")
        assert all((self.check_must, self.check_muti)) in [True, False]

        feature_list = list(range(x.shape[1]))
        fold_feature_list = self.feature_fold(feature_list)
        if self.check_must:
            fold_feature_list = [i for i in fold_feature_list if i not in self.must_unfold_add]
        slice_all = [combinations(fold_feature_list, i) for i in self.n_select]
        slice_all = [list(self.feature_unfold(_)) for i in slice_all for _ in i]

        scores = parallelize(n_jobs=self.n_jobs, func=score, iterable=slice_all)

        feature_combination = slice_all
        index = np.argmax(scores)
        select_feature = feature_combination[int(index)]
        su = np.zeros(x.shape[1], dtype=np.bool)
        su[select_feature] = 1
        self.best_score_ = max(scores)
        self.score_ = scores
        self.support_ = su
        self.estimator_ = clone(self.estimator)
        if self.refit:
            if not hasattr(self.estimator_, 'best_score_'):
                warnings.warn(UserWarning(
                    "The self.estimator_ :{} used all the X,y data.".format(self.estimator_.__class__.__name__),
                    "please be careful with the later 'score' and 'predict'."))
            if hasattr(self.estimator_, 'best_score_') and hasattr(self.estimator_, "refit") \
                    and self.estimator_.refit is True:
                warnings.warn(UserWarning(
                    "The self.estimator_ :{} used all the X,y data.".format(self.estimator_.__class__.__name__),
                    "please be careful with the later 'score' and 'predict'."))
            if hasattr(self.estimator_, "max_features"):
                self.estimator_.max_features = np.array(select_feature).shape[0]
            self.estimator_.fit(x[:, select_feature], y)
        self.n_feature_ = len(select_feature)
        self.score_ex = list(zip(feature_combination, scores))
        self.scatter = list(zip([len(i) for i in slice_all], scores))
        self.score_ex.sort(key=lambda _: _[1], reverse=True)

        return self

    @if_delegate_has_method(delegate='estimator_')
    def predict(self, X):
        """Reduce X to the selected feature and then Fit using the underlying estimator.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_


ExhaustionCV = Exhaustion

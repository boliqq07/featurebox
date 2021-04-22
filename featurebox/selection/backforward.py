# -*- coding: utf-8 -*-

# @Time   : 2019/5/23 20:40
# @Author : Administrator
# @Project : feature_preparation
# @FileName: backforward.py
# @Software: PyCharm

"""Forward_and_back feature elimination for feature ranking"""

import copy
from functools import partial
from typing import List

import numpy as np
from mgetool.tool import parallelize
from sklearn.base import BaseEstimator, is_classifier
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _score, cross_val_score
from sklearn.utils.metaestimators import if_delegate_has_method, _safe_split
from sklearn.utils.validation import check_is_fitted, check_X_y, check_random_state

from featurebox.selection.mutibase import MutiBase


def _baf_single_fit(train, test, baf, estimator, X, y, scorer, random_state):
    """"""
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    baf_i = clone(baf)
    baf_i.random_state = random_state
    baf_i._fit(X_train, y_train)
    return baf_i.support_, _score(baf_i.estimator_, baf_i.transform(X_test, ), y_test, scorer), baf_i.score_


class BackForward(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MutiBase):
    """
    BackForward method to selected features.

    estimator:
        A supervised sklearn learning estimator with a ``_fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    n_type_feature_to_select: int or None (default=None)
        The number of feature to selection. If `None`, selection the features with best score.

    n_feature_: int
        The number of selected feature.

    support_: array of shape [n_feature]
        The mask of selected feature.

    estimator_: object
        The external estimator _fit on the reduced dataset.

    Examples:
    -----------

    >>> from sklearn.datasets import load_boston
    >>> from sklearn.svm import SVR
    >>> X,y = load_boston(return_X_y=True)
    >>> svr= SVR()
    >>> bf = BackForward(svr,primary_feature=4,  random_state=1)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False, False, False, False, False, False, False, False, False,
           False,  True, False,  True])
    """

    def __init__(self, estimator: BaseEstimator, n_type_feature_to_select: int = None, primary_feature: int = None,
                 muti_grade: int = 2, muti_index: List = None,
                 must_index: List = None, tolerant: float = 0.01, verbose: int = 0, random_state: int = None):
        """

        Parameters
        ----------
        estimator:sklearn.estimator
            sklearn.estimator
        n_type_feature_to_select:int
            force select number max
        primary_feature:int
             primary features to start loop, default n_features//2.
        muti_grade: int
            group number
        muti_index:
            group index
        must_index:
            must selection index
        tolerant:
            tolerant for rank compare.
        verbose:int
            print or not
        random_state:int
            random_state
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=must_index)
        self.estimator = estimator
        self.n_type_feature_to_select = n_type_feature_to_select
        self.primary_feature = primary_feature
        self.verbose = verbose
        self.score_ = []
        self.random_state = random_state
        self.tolerant = tolerant

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the baf model and then the underlying estimator on the selected
           feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            The training input0 samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, x, y):

        def add_slice(slice10, slice20):
            best0 = score(slices=slice10)

            for add1 in list(slice20):
                # or random.choice() >= score(slice1):
                slice10.append(add1)
                test = score(slices=slice10)
                slice1_, test_ = sub_slice(slice10)
                if test_ >= max(best0, test):
                    best0 = test_
                    slice10 = slice1_
                elif test > max(best0, test_) + 2 * self.tolerant:
                    best0 = test
                else:
                    slice10.remove(add1)
                if self.verbose > 0:
                    print("Fitting estimator with {} feature {}".format(len(slice10), best0))
                self.score_.append((tuple(slice10), best0))
            return slice10

        def sub_slice(slice10):
            best0 = score(slices=slice10)
            slice0 = list(copy.deepcopy(slice10))
            slice10 = list(slice10)
            ran.shuffle(slice10)
            if self.check_must:
                slice10 = [_ for _ in slice10 if _ not in self.check_must]
            for sub in list(slice10):
                slice0.remove(sub)
                test0 = score(slices=slice0)
                if test0 > best0 - self.tolerant:
                    best0 = test0
                else:
                    slice0.append(sub)
                # print(slice0, best)
            return slice0, best0

        def sub_slice_force(slice10):
            best0 = score(slices=slice10)
            slice0 = list(copy.deepcopy(slice10))
            ran.shuffle(slice0)

            while len(slice0) > self.n_type_feature_to_select:
                slice_all = []
                slice10 = list(copy.deepcopy(slice0))
                if self.check_must:
                    slice10 = [_ for _ in slice10 if _ not in self.check_must]
                le = len(slice10)
                while le:
                    slice_all.append(copy.deepcopy(slice0))
                    le -= 1
                for slice_, sub in zip(slice_all, slice10):
                    slice_.remove(sub)
                test0 = [score(slices=i) for i in slice_all]
                index = int(np.argmax(test0))
                best0 = max(test0)
                slice0 = slice_all[index]
                # print(slice0, best0)
                if self.verbose > 0:
                    print("Fitting estimator with {} feature {}".format(len(slice0), best0))
                self.score_.append((tuple(slice0), best0))
            return slice0, best0

        def score_pri(slices, x0, y0):
            slices = list(slices)
            if len(slices) <= 1:
                score0 = - np.inf
            else:
                slices = self.feature_unfold(slices)
                data_x0 = x0[:, slices]

                self.estimator.fit(data_x0, y0)
                if hasattr(self.estimator, 'best_score_'):
                    score0 = np.mean(self.estimator.best_score_)
                else:
                    score0 = np.mean(cross_val_score(self.estimator, data_x0, y0, cv=5))
            return score0

        score = partial(score_pri, x0=x, y0=y)

        self.score_ = []
        x, y = check_X_y(x, y, "csc")
        assert all((self.check_must, self.check_muti)) in [True, False]
        # Initialization
        if self.check_muti:
            n_feature = (self.muti_index[1] - self.muti_index[0]) // self.muti_grade + self.muti_index[0]
        else:
            n_feature = x.shape[1]
        if self.primary_feature is None:
            primary_feature = n_feature // 2
        else:
            primary_feature = self.primary_feature

        feature_list = list(range(x.shape[1]))
        fold_feature_list = self.feature_fold(feature_list)

        ran = check_random_state(self.random_state)
        slice1 = ran.choice(fold_feature_list, primary_feature, replace=False)
        slice1 = list(self.feature_must_fold(slice1))
        slice2 = list(set(fold_feature_list) - set(slice1))
        ran.shuffle(slice2)

        slice1 = add_slice(slice1, slice2)
        if isinstance(self.n_type_feature_to_select, int) and len(slice1) > self.n_type_feature_to_select:
            slice1, best = sub_slice_force(slice1)

        select_feature = self.feature_unfold(slice1)
        su = np.zeros(x.shape[1], dtype=np.bool)
        su[select_feature] = 1
        self.support_ = su
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x[:, select_feature], y)
        self.n_feature_ = len(select_feature)

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected feature and then using the underlying estimator to predict.

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

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

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


class BackForwardCV(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    BackForwardCV.

    estimator : object
        A supervised learning estimator with a ``_fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.


    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

    scoring : string, callable or None, optional, (default=None)
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, (default=0)
        Controls verbosity of output.

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    n_feature_ : int
        The number of selected feature with cross-validation.

    support_ : array of shape [n_feature]
        The mask of selected feature.

    estimator_ : object
        The external estimator _fit on the reduced dataset.


    Examples:
    -----------

    >>> from sklearn.datasets import load_boston
    >>> from sklearn.svm import SVR
    >>> X,y = load_boston(return_X_y=True)
    >>> svr= SVR()
    >>> bf = BackForwardCV(svr,primary_feature=9,  random_state=1,cv=5)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False, False, False, False, False, False, False, False, False,
           False,  True, False,  True])
    >>> bf.score_
    0.656494361629943

    """

    def __init__(self, estimator: BaseEstimator, n_type_feature_to_select: int = None,
                 primary_feature: int = None, muti_grade: int = 2, muti_index: List = None,
                 must_index: List = None, verbose: int = 0, random_state: int = None,
                 cv: int = 5, scoring: str = "r2", n_jobs: int = None, refit=False):
        """

        Parameters
        ----------
        estimator:sklearn.estimator
            sklearn.estimator
        n_type_feature_to_select:int
            force select number max
        primary_feature:int
             expectation select number
        muti_grade: int
            group number
        muti_index: list
            group index
        must_index: list
            must selection index
        random_state:int
            random_state
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`sklearn.model_selection.StratifiedKFold` is used. If the
            estimator is a classifier or if ``y`` is neither binary nor multiclass,
            :class:`sklearn.model_selection.KFold` is used.

        scoring : string, callable or None, optional, (default=None)
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        verbose : int, (default=0)
            Controls verbosity of output.

        n_jobs : int or None, optional (default=None)
            Number of cores to run in parallel while fitting across folds.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
            for more details.
        refit:
            False
        """

        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_type_feature_to_select = n_type_feature_to_select
        self.primary_feature = primary_feature
        self.muti_grade = muti_grade
        self.muti_index = muti_index
        self.must_index = must_index
        self.score_ = []
        self.random_state = random_state
        self.refit = refit
        # spath.support_cv = ""
        # spath.score_cv = ""
        # spath.support_ = ""
        # spath.score_ = ""
        # spath.estimator_ = ""
        # spath.n_feature_ = ""

    def fit(self, X, y, groups=None):
        """Fit the baf model and automatically tune the number of selected feature.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_feature]
            Training vector, where `n_samples` is the number of samples and
            `n_feature` is the total number of feature.

        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like, shape = [n_samples], optional
            cal_group labels for the samples used while splitting the dataset into
            train/test set.
        """
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        ran = check_random_state(self.random_state)

        baf = BackForward(estimator=self.estimator,
                          n_type_feature_to_select=self.n_type_feature_to_select,
                          verbose=self.verbose, primary_feature=self.primary_feature,
                          muti_grade=self.muti_grade, muti_index=self.muti_index,
                          must_index=self.must_index, random_state=ran)

        func = partial(_baf_single_fit, baf=baf, estimator=self.estimator, X=X, y=y, scorer=scorer, random_state=ran)

        scores = parallelize(n_jobs=self.n_jobs, func=func, iterable=cv.split(X, y, groups), respective=True)

        support, scores, score_step = zip(*scores)
        best_support = support[np.argmax(scores)]
        best_score = max(scores)
        # Re-execute an elimination with best_k over the whole set

        # Set final attributes
        self.support_step = score_step
        self.support_cv = support
        self.support_ = best_support
        self.score_cv = scores
        self.score_ = best_score
        self.estimator_ = clone(self.estimator)
        if self.refit:
            self.estimator_.fit(X[:, self.support_], y)
        self.n_feature_ = np.count_nonzero(support)
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected feature and then Fit using the underlying estimator.

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

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

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

# from sklearn.datasets import load_boston
# from sklearn.svm import SVR
# X,y = load_boston(return_X_y=True)
# svr= SVR()
# bf = BackForwardCV(svr, primary_feature=9,  random_state=1, cv =5)
# new_x = bf.fit_transform(X,y)
# bf.support_

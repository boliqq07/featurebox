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
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, check_X_y

from featurebox.selection.multibase import MultiBase
from mgetool.tool import parallelize


class Exhaustion(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MultiBase):
    """
    Exhaustion features combination.

    Attributes
    ----------
    n_feature_: int
        The number of selected features finally.

    support_: array of shape [n_feature]
        The mask of selected features finally.

    estimator_: object
        The best model with the best features finally (refited with all data.).

    best_score_: float
        Best score of best model of best features.

    Examples
    ----------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import cross_val_predict
    >>> from sklearn.svm import SVR
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> X_train,y_train,X_test,y_test = X[:50],y[:50],X[-50:],y[-50:]

    >>> svr = SVR()
    >>> bf = Exhaustion(svr,n_select=(2,),refit=True,note=False)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False, False, False,  True, False,  True, False, False])
    >>> train_score = bf.score(X_train,y_train)  # train score
    >>> test_score = bf.score(X_test,y_test) # test score in more data.
    >>> np.mean(cross_val_score(bf.estimator_,X_train[:,bf.support_],y_train,cv=5)) # re cv_score in manually.
    -2.888471220974372
    >>> np.mean(cross_val_predict(bf.estimator_,X_train[:,bf.support_],y_train,cv=5)) # re cv_predict for plot.
    1.6001222987265382

    Examples
    ----------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.svm import SVR
    >>> from sklearn import model_selection
    >>> X,y = fetch_california_housing(return_X_y=True)
    >>> X = X[:100]
    >>> y = y[:100]
    >>> svr= SVR()

    >>> gd = model_selection.GridSearchCV(svr, param_grid=[{"C": [1, 10]}], n_jobs=1, cv=3)
    >>> bf = Exhaustion(gd,n_select=(2,),refit=True,note=False,cv=5)
    Uniform parameter in SearchCV and Exhaustion:
    (scoring=None, cv=5, refit=True)
    >>> new_x = bf.fit_transform(X,y)
    >>> bf.support_
    array([False, False, False,  True, False,  True, False, False])
    >>> bf.best_score_
    -0.7336740728050252
    """

    def __init__(self, estimator: BaseEstimator, n_select: Tuple = (2, 3, 4),
                 multi_grade: int = None, multi_index: List = None, must_index: List = None,
                 n_jobs: int = 1, refit: bool = False, cv: int = 5, scoring: str = None, note=True):
        """

        Parameters
        ----------
        estimator:
            sklearn model or GridSearchCV.
        n_select:tuple
            the n_select list,default,n_select=(3, 4).
        multi_grade:list
            binding_group size, calculate the correction between binding.
        multi_index:list
            the range of multi_grade:[min,max).
        must_index:list
            the columns force to index.
        n_jobs:int
            n_jobs.
        refit:bool
            refit or not, if refit the model would use all data.
        cv:bool
            if estimator is sklearn model, used cv, else pass.
        scoring:None,str
            scoring method name.
        note:bool
            print note or not.
        """
        super().__init__(multi_grade=multi_grade, multi_index=multi_index, must_index=must_index)
        if any((hasattr(estimator, "max_features") and refit,
                isinstance(estimator, BaseSearchCV) and hasattr(estimator.estimator, "max_features") and refit)):
            warnings.warn(
                "For estimator with 'max_features' attribute, the 'max_features' would changed with each sub-data. \n"
                "Please change and define 'max_features' by SearchCV testing after Exhaustion.\n", UserWarning)

        if refit and note:
            if isinstance(estimator, BaseSearchCV) and estimator.refit is True:
                print(
                    f"""Note:
    If refit, the self.estimator_ :{estimator.__class__.__name__} would use all the data in ``fit`` function,
    1. Be careful with the 'score' and 'predict' functions,
    Those are **training** score/predict if data in ``predict`` function not changed!
    Those are **testing** score/predict if data in ``predict`` function changed!
    2. To get CV result for evaluation:
    self.estimator_ is the SearchCV object, check 'self.estimator_.cv_result' to get CV result.
    Using 'self.best_score_' or 'self.estimator_.best_score_' for evaluation,
    Use 'cross_val_predict(self.estimator_,X[:, self.support_],y)' for plotting.""")
            else:
                print(
                    f"""Note:
    If refit, the self.estimator_ :{estimator.__class__.__name__} would use all the data in ``fit`` function,
    1. Be careful with the 'score' and 'predict' functions:
    Those are **training** score/predict, if data in ``predict`` function not changed!
    Those are **testing** score/predict, if data in ``predict`` function changed!
    2. To get CV result for evaluation:
    Use 'self.best_score_' or 'cross_val_score(self.estimator_,X[:, self.support_],y)' for evaluation,
    Use 'cross_val_predict(self.estimator_,X[:, self.support_],y)' for plotting.""")

        if cv <= 1:
            warnings.warn(
                "\nThe cv <= 1, the exhaustion would not use cross validate, and treat all data as train data, \n"
                "cv<=1 is just used for debug!!!".format(
                    estimator.__class__.__name__), UserWarning)

        if isinstance(estimator, BaseSearchCV):
            print(f"Uniform parameter in SearchCV and Exhaustion:\n"
                  f"(scoring={scoring}, cv={cv}, refit={refit})")
            estimator.scoring = scoring
            estimator.cv = cv
            estimator.n_jobs = 1
            estimator.refit = refit

        self.scoring = scoring
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
                    estimator_ = clone(self.estimator)
                    estimator_.max_features = data_x0.shape[1]

                elif isinstance(self.estimator, BaseSearchCV) and hasattr(self.estimator.estimator, "max_features"):
                    estimator_ = clone(self.estimator)
                    estimator_.estimator.max_features = data_x0.shape[1]
                else:
                    estimator_ = estimator

                if hasattr(estimator_, "best_score_") or isinstance(estimator_, BaseSearchCV):
                    estimator_.fit(data_x0, y0)
                    score0 = np.mean(estimator_.best_score_)  # score_test
                elif self.cv <= 1:
                    estimator_.fit(data_x0, y0)
                    score0 = estimator_.score(data_x0, y0, )
                else:
                    score0 = cross_val_score(estimator_, data_x0, y0, cv=self.cv, scoring=self.scoring)
                    score0 = np.mean(score0)
                # print(slices, score0)
            return score0

        score = partial(score_pri, x0=x, y0=y)

        self.score_ = []
        x, y = check_X_y(x, y, "csc")
        assert all((self.check_must, self.check_multi)) in [True, False]

        feature_list = list(range(x.shape[1]))
        fold_feature_list = self.feature_fold(feature_list)
        if self.check_must:
            fold_feature_list = [i for i in fold_feature_list if i not in self.must_unfold_add]
        slice_all = [combinations(fold_feature_list, i) for i in self.n_select]
        slice_all = [list(self.feature_unfold(_)) for i in slice_all for _ in i]
        [i.sort() for i in slice_all]

        scores = parallelize(n_jobs=self.n_jobs, func=score, iterable=slice_all)

        feature_combination = slice_all
        index = np.argmax(scores)
        select_feature = np.array(feature_combination[int(index)])
        su = np.zeros(x.shape[1], dtype=bool)
        su[select_feature] = 1
        self.best_score_ = max(scores)
        self.score_ = scores
        self.support_ = su
        self.estimator_ = clone(self.estimator)

        if self.refit:
            if hasattr(self.estimator, "max_features"):
                self.estimator_.max_features = select_feature.shape[0]
            elif isinstance(self.estimator, BaseSearchCV) and hasattr(self.estimator.estimator, "max_features"):
                self.estimator_.refit = self.refit
                self.estimator_.estimator.max_features = select_feature.shape[0]
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
    def score(self, X, y, scoring=None):
        """Reduce X to the selected feature and then return the score of the underlying estimator.
        Only available ``refit=True``.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.

        scoring : str, callable, default=None
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:
            a single string (see :ref:`scoring_parameter`)

            The score defined by ``scoring`` if provided, and the
            ``estimator_.score`` method otherwise else raise error.
        """
        check_is_fitted(self, 'estimator_')
        scoring = scoring if scoring is not None else self.scoring
        scorer = check_scoring(self.estimator_, scoring=scoring, allow_none=False)
        return scorer(self.estimator_, self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_


ExhaustionCV = Exhaustion

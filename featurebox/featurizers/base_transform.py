"""Base"""
from multiprocessing import cpu_count
from typing import List, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from mgetool.tool import parallelize
from monty.json import MSONable


class BaseFeature(MSONable):
    """
    **Using a BaseFeature Class**

    Abstract class to calculate features from ``MSONable``.
    That means you can embed this feature directly into ``BaseFeature`` class implement.
    ::

        class MatFeature(BaseFeature):
            def convert(spath, *x):
                ...

    ``BaseFeature`` implement :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin`
    that means you can use it in a scikit-learn way.
    ::

        feature = SomeFeature()
        features = feature.fit_transform(X)

    .. note::

        The ``convert`` method should be rewrite to deal with single case. And the ``transform`` and ``fit_transform``
        Will be established for list of case automatically.

    **Adding references**

    ``BaseFeature`` also provide you to retrieving proper references for a feature.
    The ``__citations__`` returns a list of papers that should be cited.
    The ``__authors__`` returns a list of people who wrote the feature.
    Also can be accessed from property ``citations`` and ``citations``.

    These operations must be implemented for each new feature:

    - ``feature_labels`` - Generates a human-meaningful x_name for each of the features. Implement this as property.

    which can be set by ``set_feature_labels``

    Also suggest to implement these two properties:

    - ``citations`` - Returns a list of citations in BibTeX format.
    - ``authors`` - Returns a list of people who contributed writing a paper.

    .. note::

        None of these operations should change the state of the feature. I.e.,
        running each method twice should no produce different results, no class
        attributes should be changed, Running one operation should not affect the
        output of another.

    """

    __authors__ = ['boliqq07']
    __citations__ = ['No citations']
    _n_jobs = 1
    _feature_labels = []

    def __init__(self, n_jobs: int = 1, *, on_errors: str = 'raise', return_type: str = 'any',
                 batch_calculate: bool = False,
                 batch_size: int = 30):
        """
        Parameters
        ----------
        batch_size: int
            size of batch.
        batch_calculate :bool
            batch_calculate or not.
        n_jobs: int
            Parallel number.
        on_errors: str
            How to handle the exceptions in a feature calculations. Can be ``nan``, ``keep``, ``raise``.
            When 'nan', return a column with np.nan. The length of column corresponding to the number of feature
            labs.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            'array' and 'df' force return type to np.ndarray and pd.DataFrame respectively.
            If 'any', without type conversion .
            Default is 'any'
        """
        self.return_type = return_type
        self.n_jobs = n_jobs
        self.on_errors = on_errors
        self._kwargs = {}
        self.support_ = []
        self.d2 = False
        self._feature_labels = []
        self.batch_calculate = batch_calculate
        self.batch_size = batch_size
        import inspect
        inspect.getfullargspec(self.convert)

    @property
    def n_jobs(self):
        """
        n_jobs: int
            Parallel number.
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        """Set the number of threads for this"""
        if n_jobs > cpu_count() or n_jobs == -1:
            self._n_jobs = cpu_count()
        else:
            self._n_jobs = n_jobs

    def fit(self, *args, **kwargs):
        """fit function in :class:`BaseFeature` are weakened and just pass parameter."""
        _ = args
        self._kwargs = kwargs
        return self

    def fit_transform(self, X: List, y=None, **kwargs) -> Any:
        """
        If `convert` takes multiple inputs, supply inputs as a list of tuples.

        Copy from Mixin class for all transformers in scikit-learn. TransformerMixin

        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : list
            list of case.
        y : None
            deprecated.
        **kwargs : dict
            Additional fit or transform parameters.

        Returns
        --------
        X_new :
            result data.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **kwargs).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **kwargs).transform(X)

    def transform(self, entries: List) -> Any:
        """
        transform a list of entries.
        If `convert` takes multiple inputs, supply inputs as a list of tuples.

        Parameters
        ----------
        entries: list
            A list of entries to be featured.

        Returns
        ---------
        DataFrame:
            features for each entry.
        """

        # Check inputs
        if not isinstance(entries, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        # Special case: Empty list
        if len(entries) == 0:
            return []

        # Run the actual feature
        rets = parallelize(self.n_jobs, self._wrapper, entries, tq=True)

        ret, self.support_ = zip(*rets)

        try:
            labels_len = len(self.feature_labels)
            if labels_len > 0:
                labels = self.feature_labels
            else:
                labels = None
        except (NotImplementedError, TypeError):
            labels = None

        if self.return_type == 'any':
            return ret

        if self.return_type == 'array':
            return np.array(ret)

        if self.return_type == 'df':
            if isinstance(entries, (pd.Series, pd.DataFrame)):
                return pd.DataFrame(ret, index=entries.index, columns=labels)
            return pd.DataFrame(ret, columns=labels)

    def _wrapper(self, x):
        """
        An exception wrapper for feature, used in feature_many and
        feature_dataframe. feature_wrapper changes the behavior of feature
        when ignore_errors is True in feature_many/dataframe.

        Parameters
        ----------
        x:
            input0 data_cluster to feature (type depends on feature).

        Returns
        -------
        result:tuple
            The first is calculated result, and the second is bool to mark data availability.
        """

        try:
            if isinstance(x, tuple):
                try:
                    con = self.convert(*x)
                except TypeError as e:
                    print(e)
                    raise TypeError("The tuple parameter would be separate automatically,"
                                    "if you case data is tuple, and Pass it to the first argument,"
                                    "please change it to list")
            else:
                try:
                    con = self.convert(x)
                except TypeError as e:
                    print(e)
                    raise TypeError("Please check the above errors ,"
                                    "If the parameters should be unpack, please change the case to tuple, "
                                    "the other type(like list) would just pass to the first parameter integrally")

            if isinstance(con, (List, Tuple)):
                if len(con) == 2 and isinstance(con[1], bool):
                    pass
                else:
                    con = (con, True)
            else:
                con = (con, True)
            return con

        except BaseException as e:
            if self.on_errors == "nan":
                print("Bad conversion for:", x)
                return np.nan, False
            elif self.on_errors == "raise":
                raise e

    def convert(self, d):
        """
        Main feature function, which has to be implemented
        in any derived feature subclass.

        Parameters
        -----------
        d:
            input data_cluster to feature.

        Returns
        -------
        new_x: np.ndarray
            new x.
        conversion:bool
            Whether the conversion is perfect.
        """

        if isinstance(d, np.ndarray):
            if d.ndim == 1:
                return np.array(self._convert(d))
            if d.ndim == 2:
                if self.d2:
                    return np.array(self._convert(d))
                else:
                    return np.array([self._convert(i) for i in d])
            if d.ndim == 3:
                return np.array([[self._convert(i) for i in di] for di in d])

        try:
            return np.array(self._convert(d))
        except BaseException as e:
            print(e)
            raise ValueError("Error when try to convert: \n {}".format(str(d)))

    @property
    def feature_labels(self):
        """Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        return self._feature_labels

    def set_feature_labels(self, values: List[str]):
        """Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        self._feature_labels = values

    @property
    def citations(self):
        """Citation(s) and reference(s) for this feature.

        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """
        return '\n'.join(self.__citations__)

    @property
    def authors(self):
        """List of implementors of the feature.

        Returns:
            (list) each element should either be a string with author x_name (e.g.,
                "Anubhav Jain") or a dictionary  with required key "x_name" and other
                keys like "email" or "institution" (e.g., {"x_name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        return '\n'.join(self.__authors__)

    def _convert(self, d: np.ndarray) -> np.ndarray:
        """1d array input"""
        return d

    def __add__(self, other):
        raise TypeError("This method has no add")


Converter = BaseFeature  # old name


class DummyConverter(Converter):
    """
    Dummy converter as a placeholder, Do nothing.
    """

    def convert(self, d) -> np.ndarray:
        """
        Dummy convert, does nothing to input.

        Args:
            d (Any): input object

        Returns: d

        """
        d = np.array(d)

        return d.reshape(-1, 1) if d.ndim == 1 else d

    def _convert(self, d: np.ndarray) -> np.ndarray:
        pass


class ConverterCat(BaseFeature):
    """Pack the converters in to one unified approach.
    The same type Converter would merged and different would order to run.
    Thus, keeping the same type is next to each other!

    Examples
    ----------
    >>> tmps = ConverterCat(
    ...    AtomEmbeddingMap(),
    ...    AtomEmbeddingMap("ie.json")
    ...    AtomTableMap(search_tp="name"))
    >>> tmp.convert(x)

    """

    def __init__(self, *args: Converter, n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'any'):
        """

        Parameters
        ----------
        args: Converter
            List of Converter
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.args = self.sums(list(args))

    @staticmethod
    def sums(args):
        """SUM"""
        i = 0
        while i < len(args) - 1:
            try:
                args[i] = args[i] + args[i + 1]
                del args[i + 1]
            except (AssertionError, TypeError, IndexError, NotImplementedError):
                i += 1
        return args

    def convert(self, d):
        """convert batched"""
        data = []
        for ci in self.args:
            data.append(ci.convert(d))
        else:
            try:
                return np.concatenate(data, axis=-1)
            except ValueError:
                return data

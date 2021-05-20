"""
Operation utilities on lists and arrays
"""
from collections import Iterable
from typing import Union, List, Sequence, Optional

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


def to_list(x: Union[Iterable, np.ndarray]) -> List:
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, Iterable):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # noqa
    return [x]


def fast_label_binarize(value: List, labels: List) -> List[int]:
    """Faster version of label binarize

    `label_binarize` from scikit-learn is slow when run 1 label at a time.
    `label_binarize` also is efficient for large numbers of classes, which is not
    common in `megnet`

    Args:
        value: Value to encode
        labels (list): Possible class values
    Returns:
        ([int]): List of integers
    """

    if len(labels) == 2:
        return [int(value == labels[0])]
    output = [0] * len(labels)
    if value in labels:
        output[labels.index(value)] = 1
    return output


def check_shape(array: Optional[np.ndarray], shape: Sequence) -> bool:
    """
    Check if array complies with shape. Shape is a sequence of
    integer that may end with None. If None is at the end of shape,
    then any shapes in array after that dimension will match with shape.

    Example: array with shape [10, 20, 30, 40] matches with [10, 20, None], but
        does not match with shape [10, 20, 30, 20]

    Args:
        array (np.ndarray or None): array to be checked
        shape (Sequence): integer array shape, it may ends with None
    Returns: bool
    """
    if array is None:
        return True
    if all(i is None for i in shape):
        return True

    array_shape = array.shape
    valid_dims = [i for i in shape if i is not None]
    n_for_check = len(valid_dims)
    return all(i == j for i, j in zip(array_shape[:n_for_check], valid_dims))


def train_test(*arrays, **options):

    """Split arrays or matrices into random train and test subsets

    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)


    """

    train_test_data = train_test_split(*arrays, **options)
    le = len(train_test_data)
    (*X_train, y_train), (*X_test, y_test) = [train_test_data[i] for i in range(le) if i % 2 == 0], [train_test_data[i]
                                                                                                     for i in range(le)
                                                                                                     if i % 2 == 1]
    return X_train, y_train, X_test, y_test

"""
Operation utilities on lists and arrays
"""
from collections import Iterable
from typing import Union, List, Sequence, Optional

import numpy as np
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
    """Just return train and test, respectively"""
    train_test_data = train_test_split(*arrays, **options)
    le = len(train_test_data)
    return [train_test_data[i] for i in range(le) if i % 2 == 0], [train_test_data[i] for i in range(le) if i % 2 == 1]

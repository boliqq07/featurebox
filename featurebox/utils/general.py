"""
Operation utilities on lists and arrays, for torch,

Notes
    just for network.
"""
from collections import abc
from functools import wraps
from typing import Union, List, Sequence, Optional

import numpy as np
# import torch
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Module


def to_list(x: Union[abc.Iterable, np.ndarray]) -> List:
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, abc.Iterable):
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


def train_test_pack(*arrays, out=0, **options):
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

    out:int
        pack the pack to format,
        pack=None: (0:n-1)
        pack=0: (0:n-1:2, 1:n:2)
        pack=1: (0:n-3:2, -2, 1:n-2:2, -1)
        pack=2: (0:n-5:2, -4, -2, 1:n-4:2, -3, -1)

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = train_test_split(
    ...     *X, y, test_size=0.33, random_state=42)

    """
    num = len(arrays)
    assert num >= out

    train_test_data = train_test_split(*arrays, **options)
    le = len(train_test_data)
    if out == 0:
        X_train = [train_test_data[i] for i in range(le) if i % 2 == 0]
        X_test = [train_test_data[i] for i in range(le) if i % 2 == 1]
        if num == 1:
            X_train, X_test = X_train[0], X_test[0]
        return X_train, X_test
    elif out == 1:
        (*X_train, y_train) = [train_test_data[i] for i in range(le) if i % 2 == 0]
        (*X_test, y_test) = [train_test_data[i] for i in range(le) if i % 2 == 1]
        if num == 2:
            X_train, X_test = X_train[0], X_test[0]
        return X_train, y_train, X_test, y_test
    elif out == 2:
        (*X_train, y_train, label_train) = [train_test_data[i] for i in range(le) if i % 2 == 0]
        (*X_test, y_test, label_test) = [train_test_data[i] for i in range(le) if i % 2 == 1]
        if num == 3:
            X_train, X_test = X_train[0], X_test[0]
        return X_train, y_train, X_test, y_test, label_train, label_test
    else:
        return train_test_data


def re_pbc(pbc: Union[bool, List[bool], np.ndarray], return_type="bool"):
    if pbc is True:
        pbc = [1, 1, 1]
    elif pbc is False:
        pbc = [0, 0, 0]
    elif isinstance(pbc, abc.Iterable):
        pbc = [1 if i == True or i == 1 else 0 for i in pbc]
    else:
        raise TypeError("Can't accept {}".format(pbc))
    if return_type == "bool":
        pbc = np.array(pbc) == 1
    else:
        pbc = np.array(pbc)
    return pbc


# a = re_pbc(np.array([True,True,False]), return_type="int")

def getter_arr(obj, pi):
    """Get prop.
    """
    if "." in pi:
        pis = list(pi.split("."))
        pis.reverse()
        while len(pis):
            s = pis.pop()
            obj = getter_arr(obj, s)
        return obj
    elif "()" in pi:
        return getattr(obj, pi[:-2])()

    else:
        return getattr(obj, pi)


def temp_jump(mark=0, temp_device=None, old_device=None, temp=True, back=True):
    def f(func):
        return _temp_jump(func, mark=mark, temp_device=temp_device, old_device=old_device, temp=temp, back=back)

    return f


def temp_jump_cpu(mark=0, temp_device="cpu", old_device=None, temp=True, back=True):
    def f(func):
        return _temp_jump(func, mark=mark, temp_device=temp_device, old_device=old_device, temp=temp, back=back)

    return f


def _temp_jump(func, mark=0, temp_device="cpu", old_device=None, temp=True, back=True):
    """temp to cpu to calculate and re-back the init device data."""

    @wraps(func)
    def wrapper(*args, **kwargs):

        if temp_device is None:
            device = args[0].device
        else:
            device = torch.device(temp_device) if isinstance(temp_device, str) else temp_device

        if old_device is None:
            device2 = args[mark + 1].device if len(args) > 1 else list(kwargs.values())[0].device
        else:
            device2 = torch.device(old_device) if isinstance(old_device, str) else old_device

        if temp:
            args2 = [args[0]]
            for i in args[1:]:
                try:
                    args2.append(i.to(device=device, copy=False))
                except AttributeError:
                    args2.append(i)
            kwargs2 = {}
            for k, v in kwargs.items():
                try:
                    if isinstance(v, tuple):
                        kwargs2[k] = [i.to(device=device, copy=False) for i in v]
                    else:
                        kwargs2[k] = v.to(device=device, copy=False)
                except AttributeError:
                    kwargs2[k] = v

            result = func(*args2, **kwargs2)
        else:
            result = func(*args, **kwargs)

        if back:
            if isinstance(result, tuple):
                result2 = []
                for i in result:
                    try:
                        result2.append(i.to(device=device2, copy=False))
                    except AttributeError:
                        result2.append(i)
            else:
                try:
                    result2 = result.to(device=device2, copy=False)
                except AttributeError:
                    result2 = result
            return result2
        else:
            return result

    return wrapper


def check_device(mode: Module):
    device = _check_device(mode)
    return torch.device("cpu") if device is None else device


def _check_device(mode: Module):
    device = None
    for i in mode.children():
        if hasattr(i, "weight") and isinstance(i, Tensor):
            device = i.weight.device
            break
        elif hasattr(i, "bias"):
            device = i.bias.device
            break
        elif len(i.children()) > 0:
            device = check_device(i)
            if device is not None:
                break
    return device

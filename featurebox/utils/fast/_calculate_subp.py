import numba
import numpy as np


def subp_py(entry, mapper):
    """in place"""
    for k, i in enumerate(mapper):
        entry[np.where(entry == k)] = i
    return entry


@numba.jit(forceobj=True, signature_or_function="int64[:](int64[:],int64[:])")
def subp_numba(entry, mapper):
    """in place, departed!!!"""
    for k, i in enumerate(mapper):
        entry[np.where(entry == k)] = i
    return entry


@numba.jit(signature_or_function="int64[:](int64[:],int64[:])", nopython=True)
def subp_numba2(entry, mapper):
    """in place"""
    size = len(entry)
    for i in range(size):
        entry[i] = mapper[entry[i]]
    return entry


def subp_numba2d(entry, mapper):
    """in place"""
    shape = entry.shape
    entry = entry.ravel()
    entry = subp_numba2(entry, mapper)
    entry = entry.reshape(shape)
    return entry

# a = [np.arange(0,50)]*2000
# a = np.concatenate(a,axis=0)
# from mgetool.tool import tt
# # a = a.reshape(50,100)
# b = np.arange(0,50)
# np.random.shuffle(b)
# tt.t
# new = subp_py(a,b)
# tt.t
# new = subp_numba(a,b)
# tt.t
# new = subp_numba(a,b)
# tt.t
# new = subp_numba2(a,b)
# tt.t
# new = subp_numba2(a,b)
# tt.t
# tt.p

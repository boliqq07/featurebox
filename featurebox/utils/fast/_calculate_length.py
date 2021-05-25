import numba
import numpy as np


def cal_length_py(ax):
    # ax is list of int
    ls = []
    temp = ax[0]
    li = -1

    for ai in range(len(ax)):
        if temp == ax[ai]:
            li += 1
        else:
            temp = ax[ai]
            li += 1
            ls.append(li)
            li = 0
    ls.append(li + 1)
    ls = np.array(ls)
    return ls


@numba.jit(nopython=True, signature_or_function=["int64[:](int64[:])", "int32[:](int32[:])"])
def cal_length_numba(ax):
    # ax is list of int
    ls = []
    temp = ax[0]
    li = -1

    for ai in ax:
        if temp == ai:
            li += 1
        else:
            temp = ai
            li += 1
            ls.append(li)
            li = 0
    ls.append(li + 1)
    ls = np.array(ls)
    return ls

# a = np.arange(10000).reshape(-1,1)
# b= np.concatenate((a,a),axis=1)
# b = b.ravel()
#
# from mgetool.tool import tt
# tt.t
# c = cal_length_py(b)
# tt.t
# c = cal_length_numba(b)
# tt.t
# c = cal_length_numba(b)
# tt.t
# tt.p

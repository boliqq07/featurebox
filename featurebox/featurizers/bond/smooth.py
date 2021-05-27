from math import cos, pi
from pathlib import Path

import numba
import numpy as np
from mgetool.tool import tt

from featurebox.featurizers.base_transform import BaseFeature

MODULE_DIR = Path(__file__).parent.parent.parent.absolute()


# ###############bond##########################################

class Smooth(BaseFeature):
    """
    smooth.
    """

    def __init__(self, r_c=5, r_cs=3):
        """

        Args:
            r_cs: float, strat of decrease cutoff radius.
            r_c: (float) cutoff radius.
        """
        self.r_c = r_c
        self.r_cs = r_cs

        super(Smooth, self).__init__()

    def _convert(self, d: np.ndarray) -> np.ndarray:
        """
        d: np.ndarray, shape (N, fill_size, atom_fea_len).
            ``nbr_fea`` neighbor features for each center_index.
        """
        d = np.array(d)
        d = d.astype(np.float32)
        return smooth_func(d, self.r_c, self.r_cs)


@numba.jit("float32[:](float32[:],float32,float32)")
def _smooth_func(rs, r_c, r_cs):
    ks = []
    for r in rs:
        if r <= r_cs:
            ks.append(1 / r)
        elif r_cs < r <= r_c:
            ks.append(1 / r * (0.5 * cos((r - r_cs) / (r_c - r_cs) * pi) + 0.5))
        else:
            ks.append(0.0)
    return np.array(ks)


def smooth_func(d, r_c, r_cs):
    if d.ndim == 3:
        r = (d[:, :, 1] ** 2 + d[:, :, 2] ** 2 + d[:, :, 3] ** 2) ** 0.5
        k = np.array([_smooth_func(ri, r_c, r_cs) for ri in r])

    elif d.ndim == 2:
        r = (d[:, 1] ** 2 + d[:, 2] ** 2 + d[:, 3] ** 2) ** 0.5
        k = _smooth_func(r, r_c, r_cs)
    else:
        r = (d[1] ** 2 + d[2] ** 2 + d[3] ** 2) ** 0.5
        if r <= r_cs:
            k = 1 / r
        elif r_cs < r <= r_c:
            k = 1 / r * (0.5 * cos((r - r_cs) / (r_c - r_cs) * pi) + 0.5)
        else:
            k = 0.0

    return k * d / r
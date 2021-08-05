"""Use descriptors form ``pyXtal_FF``, in :mod:`featurebox.test_featurizers.descriptors`
"""
from typing import Dict

import numpy as np

from featurebox.featurizers.descriptors.ACSF import ACSF
from featurebox.featurizers.descriptors.EAD import EAD
from featurebox.featurizers.descriptors.EAMD import EAMD
from featurebox.featurizers.descriptors.SO3 import SO3
from featurebox.featurizers.descriptors.SO4 import SO4_Bispectrum
from featurebox.featurizers.descriptors.SOAP import SOAP
from featurebox.featurizers.descriptors.behlerparrinello import BehlerParrinello
from featurebox.featurizers.descriptors.wACSF import wACSF
from featurebox.utils.look_json import mark_classes

DesDict = mark_classes([
    ACSF,
    BehlerParrinello,
    EAD,
    EAMD,
    SOAP,
    SO3,
    SO4_Bispectrum,
    wACSF,
])

for i, j in DesDict.items():
    locals()[i] = j


def universe_refine_des(d: Dict, fill_size=10, **kwargs):
    """
    Change each center atoms has fill_size neighbors.
    More neighbors would be abandoned.
    Insufficient neighbors would be duplicated.

    # todo There is a problem with deleting atoms

    Args:
        d: dict, dict of descriptor. at lest contain "x" and "dxdr"
        fill_size: int, unstable.

    Returns:
        (center_indices,center_prop, neighbor_indices, images, distances)\n
        center_indices: np.ndarray 1d(N,).\n
        neighbor_indices: np.ndarray 2d(N,fill_size).\n
        images: np.ndarray 2d(N,fill_size,l).\n
        distance: np.ndarray 2d(N,fill_size), None.
        center_prop: np.ndarray 1d(N,l_c).\n

    """
    center, dxdr, seq = d["x"], d["dxdr"], d.get("seq")
    atom_len = center.shape[0]
    dxdr_len = dxdr.shape[0]

    if dxdr.ndim == 3:
        # please no cut
        # ACSF no
        # BehlerParrinello no
        # EAD no
        # EMAD
        # wACSF

        seq0 = seq[:, 0]

        uni = [len(seq[np.where(seq0 == i)]) for i in range(np.max(seq0) + 1)]
        if len(set(uni)) == 1:
            fill_size1 = int(dxdr_len / atom_len)
            dxdr_new = np.reshape(dxdr, (atom_len, fill_size1, -1))
            seq_len = seq.shape[0]
            fill_size2 = int(seq_len / atom_len)
            # assert fill_size2==fill_size1
            seq_new = np.reshape(seq[:, 1], (atom_len, fill_size2))
            left = fill_size - uni[0]
            if left == 0:
                pass
            elif left > 0:
                seq_new = np.concatenate((seq_new, np.array([seq_new[:, -1]] * left).T), axis=1)
                dxdr_new = np.concatenate((dxdr_new, np.array([dxdr_new[:, -1]] * left).transpose(1, 0, 2)), axis=1)

            else:
                seq_new = seq_new[:, :fill_size]
                dxdr_new = dxdr_new[:, :fill_size]

        else:
            seq_split = [list(seq[:, 1][np.where(seq0 == i)]) for i in range(np.max(seq0) + 1)]

            left = [fill_size - i for i in uni]
            seq_new = []
            dxdr_new = []
            for lefti, seqsi in zip(left, seq_split):
                if lefti == 0:
                    dxdri = dxdr[seqsi]
                elif lefti > 0:
                    seqsi.extend([seqsi[-1]] * lefti)
                    dxdri = dxdr[seqsi]
                else:
                    seqsi = seqsi[:fill_size]  # ? it is not loosely
                    dxdri = dxdr[:fill_size]

                seq_new.append(seqsi)
                dxdr_new.append(dxdri)
            seq_new = np.array(seq_new)
            dxdr_new = np.array(dxdr_new)

    elif dxdr.ndim == 4:
        # please no cut
        # soap
        # wACSF
        dxdr_new = np.reshape(dxdr, (dxdr.shape[0], dxdr.shape[1], -1))
        seq_new = np.repeat(np.arange(atom_len).reshape(-1, 1), dxdr.shape[1], axis=1).T
        left = fill_size - atom_len
        if left == 0:
            pass
        elif left > 0:
            seq_new = np.concatenate((seq_new, np.array([seq_new[:, -1]] * left).T), axis=1)
            dxdr_new = np.concatenate((dxdr_new, np.array([dxdr_new[:, -1]] * left).transpose(1, 0, 2)), axis=1)

        else:
            seq_new = seq_new[:, :fill_size]
            dxdr_new = dxdr_new[:, :fill_size]
    else:
        raise NotImplementedError("Unrecognized data format")

    assert seq_new.shape[0] == center.shape[0]
    assert seq_new.shape[1] == dxdr_new.shape[1]
    return np.array(range(atom_len)), seq_new, dxdr_new, None, center

import argparse
import os

# Due to the pymatgen is incorrect of band gap with 2 spin. we use vaspkit for extract data.
import pandas as pd
from mgetool.imports import BatchFile
import os


def from_vaspkit(d, result_name="D_BAND_CENTER", cmd="vaspkit -task 503", postprocessing=lambda x:x):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    try:
        old = os.getcwd()
        os.chdir(d)
        os.system(cmd)

        with open(result_name, mode="r") as f:
            ress = f.readlines()

        result = postprocessing(ress)

        os.chdir(old)

        return result
    except BaseException as e:
        print(e)
        print("Error for:", d)
        return {}


def from_vaspkit_all(d, repeat=0):
    data_all = {}
    for di in d:
        res = from_vaspkit(di)
        if repeat <=1 :
            data_all.update({di: res})
        else:
            for i in range(repeat):
                data_all.update({str(di)+f"S{i}": res})

    return data_all
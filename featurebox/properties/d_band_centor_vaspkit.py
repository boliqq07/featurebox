import argparse
import os

# Due to the pymatgen is incorrect of band gap with 2 spin. we use vaspkit for extract data.
import pandas as pd
from mgetool.imports import BatchFile
import os

import numpy as np
import pandas as pd
# LORBIT=2


def dbc_vaspkit(d,result_name = "D_BAND_CENTER"):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    try:
        cmd = "vaspkit -task 503"
        old = os.getcwd()
        os.chdir(d)
        os.system(cmd)

        with open(result_name, mode="r") as f:
            ress = f.readlines()

        res = []
        for i in ress:
            if i == "\n":
                break
            else:
                i = i.split(" ")
                i = [i for i in i if i != ""]
                i = [i[:-2] if "\n" in i else i for i in i]
                res.append(i)

        res = np.array(res[1:])
        result = pd.DataFrame(res,
                           columns=["Atom", "d-Band-Center (UP)", "d-Band-Center (DOWN)", "d-Band-Center (Average)"])

        os.chdir(old)

        return result
    except BaseException as e:
        print(e)
        print("Error for:", d)
        return {}


def dbc_vaspkit_all(d, repeat=0):
    data_all = {}
    for di in d:
        res = dbc_vaspkit(di)
        if repeat <=1:
            data_all.update({di: res})
        else:
            for i in range(repeat):
                data_all.update({str(di)+f"S{i}": res})

    return data_all


def main():
    """

    Example:

        $ python d_band_centor_vaspkit.py -p /home/dir_name -if POSCAR
    """
    parser = argparse.ArgumentParser(description="Get d band centor.")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-t', '--job_type', type=str, default='s')
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
    # parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default="pure_u_static")
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
    parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=3)
    args = parser.parse_args()

    if args.job_type in ["S", "s"]:
        res = dbc_vaspkit(args.path_name)
        print(args.dir_name, res)
    else:
        assert args.job_type in ["M", "m"]
        bf = BatchFile(args.path, suffix=args.suffix)
        bf.filter_dir_name(include=args.dir_include, exclude=args.dir_exclude, layer=args.layer)
        bf.filter_file_name(include=args.file_include, exclude=args.file_exclude)
        bf.merge()

        fdir = bf.file_dir
        fdir.sort()

        if not args.abspath:
            os.chdir(args.path)
            absp = os.path.abspath(args.path)
            fdir = [i.replace(absp, ".") for i in fdir]

        dbc_vaspkit_all(fdir,repeat=args.repeat)

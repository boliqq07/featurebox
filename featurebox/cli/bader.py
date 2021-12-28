import argparse
import os

# Due to the pymatgen is incorrect of band gap with 2 spin. we use vaspkit for extract data.
import pandas as pd
import pymatgen.io.vasp
from mgetool.imports import BatchFile
import os

import numpy as np

# 1
# LAECHG=.TRUE.
# LCHARG = .TRUE.
# NSW = 0
# IBRION = -1 (前面有了NSW = 0, 这个也可以不设置)

# 2
# chmod u+x ~/bin/chgsum.pl
# chmod u+x ~/bin/bader
from pymatgen.io.vasp import Potcar, Poscar


def bader(d, ):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    try:

        cmd = "chgsum.pl AECCAR0 AECCAR2"
        cmd2 = "bader CHGCAR -ref CHGCAR_sum"
        old = os.getcwd()
        os.chdir(d)
        os.system(cmd)
        os.system(cmd2)

        if os.path.isfile("POTCAR") and os.path.isfile("POSCAR"):
            potcar = Potcar.from_file("POTCAR")
            poscar = Poscar.from_file("POSCAR").site_symbols
            # todo
            result = []

        else:
            result =  []

        os.chdir(old)

        return result
    except BaseException as e:
        print(e)
        print("Error for:", d)
        return []


def bader_all(d, repeat=0,store=False):
    data_all = []
    for di in d:
        res = bader(di)
        if repeat <=1:
            data_all.append(res.values)
        else:
            for i in range(repeat):
                res2 = res.values
                res2[:,0] = res2[:,0]+f"S{i}"
                data_all.append(res2)
    data_all = np.concatenate(data_all,axis=0)
    result = pd.DataFrame(data_all,
                          columns=["File", "Atom Rank", "Symbol", "CHARGE", "Bader",
                                  ])

    if store:
        result.to_csv("bader_all_file.csv")

    return data_all

def run(args,parser):
    if args.job_type in ["S", "s"]:
        res = bader(args.path_name)
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

        bader_all(fdir, repeat=args.repeat,store=True)


class CLICommand:

    """
    批量提取 d band centor，保存到当前文件夹。 查看参数帮助使用 -h。
    首先，请确保运算子文件夹包含应有文件。

    如果在 featurebox 中运行:

    Example:

        $ featurebox bader -p /home/dir_name -if POSCAR


    cmd 命令用单引号。换行使用\n,并且其后不要留有空格。
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-t', '--job_type', type=str, default='s')
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default="pure_static")
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=True)
        parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=0)

    @staticmethod
    def run(args, parser):
        run(args,parser)


if __name__ == '__main__':
    """
    Example:

        $ python bader.py -p /home/dir_name -if POSCAR
    """
    parser = argparse.ArgumentParser(description="Get d band centor.")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-t', '--job_type', type=str, default='s')
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
    # parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default="pure_static")
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=True)
    parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=3)
    args = parser.parse_args()



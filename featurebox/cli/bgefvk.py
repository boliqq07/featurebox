import argparse
import os

# Due to the pymatgen is incorrect of band gap with 2 spin. we use vaspkit for extract data.

import pandas as pd
from mgetool.imports import BatchFile


def band_gap_from_vaspkit(d):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    try:
        cmd = "vaspkit -task 911"
        old = os.getcwd()
        os.chdir(d)
        os.system(cmd)

        with open("BAND_GAP") as f:
            res = f.readlines()

        res = [i for i in res if "(eV)" in i]

        name = [i.split(":")[0].replace("  ", "") for i in res]
        name = [i[1:] if i[0] == " " else i for i in name]
        value = [float(i.split(" ")[-1].replace("\n", "")) for i in res]
        result = {}
        for ni, vi in zip(name, value):
            result.update({ni: vi})

        os.chdir(old)

        return result
    except BaseException as e:
        print(e)
        print("Error for:", d)
        return {}


def band_gap_from_vaspkit_all(d, repeat=0):
    data_all = {}
    for di in d:
        res = band_gap_from_vaspkit(di)
        if repeat <=1:
            data_all.update({di: res})
        else:
            for i in range(repeat):
                data_all.update({str(di)+f"S{i}": res})
    data_all_pd = pd.DataFrame.from_dict(data_all).T
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_all_pd.to_csv("band_gap.csv")
    print("The data are stored in '{}'".format(os.getcwd()))
    return data_all

def run(args,parser):

    if args.job_type in ["S", "s"]:
        res = band_gap_from_vaspkit(args.path_name)
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

        band_gap_from_vaspkit_all(fdir,repeat=args.repeat)


class CLICommand:

    """
    批量提取带隙,费米能级， 查看参数帮助使用 -h。
    首先，请确保运算子文件夹包含应有文件。

    如果在 featurebox 中运行:

    Example:

        $ featurebox bgefvk -p /home/dir_name -if POSCAR


    cmd 命令用单引号。换行使用\n,并且其后不要留有空格。
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-t', '--job_type', type=str, default='s')
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default="pure_u_static")
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
        parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=0)

    @staticmethod
    def run(args, parser):
        run(args,parser)


if __name__ == '__main__':
    """
    Example:
        
        $ python bgefvk.py -p /home/dir_name -if POSCAR
    """
    parser = argparse.ArgumentParser(description="Get band gaps. Examples：\n"
                                                 "python bgefvk.py -p /home/dir_name -if POSCAR")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-t', '--job_type', type=str, default='s')
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default="pure_u_static")
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
    parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=0)
    args = parser.parse_args()
    run(args, parser)



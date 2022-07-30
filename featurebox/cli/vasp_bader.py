# -*- coding: utf-8 -*-

# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import warnings
from abc import abstractmethod
from path import Path
from pymatgen.io.vasp import Potcar, Poscar
from tqdm import tqdm
from typing import Union, Callable, List, Any


class _BasePathOut:
    """
    The subclass should fill the follow properties and functions.

    self.necessary_files = []  # fill
    self.out_file = ""  # fill
    self.software = []  # fill
    self.inter_temp_files = []  # fill
    self.key_help = []  # fill


    def run(self, path: Path, files: List = None):
        ...  # fill

    def batch_after_treatment(self, paths, res_code):
        ...  # fill

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, log=True):
        self.n_jobs = n_jobs
        self.tq = tq
        self.log = log
        self.store_single = store_single
        self.necessary_files = []  # fill
        self.out_file = ""  # fill
        self.software = []  # fill
        self.inter_temp_files = []  # fill
        self.key_help = ""  # fill

    def __str__(self):
        return f"{self.__class__.__name__} <necessary file: {self.necessary_files}, out file: {self.out_file}>"

    def __repr__(self):
        return f"{self.__class__.__name__} <necessary file: {self.necessary_files}, out file: {self.out_file}>"

    def _to_path(self, path: Union[os.PathLike, Path, pathlib.Path, str]) -> Path:
        if isinstance(path, str):
            path = path.replace("\n", "")
        return Path(path).abspath() if not isinstance(path, Path) else path

    def convert(self, path: Union[os.PathLike, Path, pathlib.Path, str]):
        self.check_software()
        path = self._to_path(path)
        path_bool = self.check_path_and_file(path)
        if path_bool:
            return self.wrapper(path)
        else:
            raise FileNotFoundError

    def transform(self, paths: List[Union[os.PathLike, Path, pathlib.Path, str]]):
        self.check_software()
        paths = [self._to_path(pi) for pi in paths]
        paths_bool = [self.check_path_and_file(pi) for pi in paths]
        if self.n_jobs == 1:
            res_code = [self.wrapper(pi) if pib is True else None for pi, pib in zip(paths, paths_bool)]
        else:
            # trues_index = np.where(np.array(paths_bool))[0]
            trues = [i for i, b in zip(paths, paths_bool) if b]
            res_code = self.parallelize_imap(self.wrapper, iterable=list(trues), n_jobs=self.n_jobs, tq=self.tq)

        return self.batch_after_treatment(paths, res_code)

    @staticmethod
    def parallelize_imap(func: Callable, iterable: List, n_jobs: int = 1, tq: bool = True):
        """
        Parallelize the function for iterable.
        """
        pool = multiprocessing.Pool(processes=n_jobs)
        if tq:
            result_list_tqdm = [result for result in tqdm(pool.imap(func=func, iterable=iterable),
                                                          total=len(iterable))]
            pool.close()
        else:
            result_list_tqdm = [result for result in pool.imap(func=func, iterable=iterable)]
            pool.close()

        return result_list_tqdm

    def check_path_and_file(self, path: Path):
        """1.Check the path exist, and necessary file."""
        if not path.exists():
            return False
        else:
            if path.isdir():
                if all([True for i in self.necessary_files if (path / i).exists()]):
                    return True
                else:
                    warnings.warn(f"Loss necessary files for {path}.")
                    return False
            elif path.isfile():
                warnings.warn("If path is one file, we think the necessary_files is just this file, "
                              "the ``necessary_files`` would be empty,"
                              "and with no follow-up necessary file inspection.")
                self.necessary_files = []
                return True
            else:
                return False

    def check_software(self):
        """2.Check software exist (just for linux). """
        for si in self.software:
            res = os.popen(f"which {si}").readlines()
            if len(res) == 0:
                raise NotImplementedError(f"Can't find the {si}, please:\n"
                                          f"1. Add this software to your PATH for system,\n"
                                          f"   such as, add 'export PATH=$PATH:~/bin/chgsum.pl' to '~/.bashrc' ;\n"
                                          f"2. keep access permission by such as, 'chmod u+x ~/bin/chgsum.pl' .")

    def wrapper(self, path: Path):
        """For exception."""
        if path.isfile():
            files = [path, ]  # 单个文件
        else:
            files = [path / i for i in self.necessary_files]

        old = os.getcwd()
        os.chdir(path)
        try:
            result = self.run(path, files)
            if result is None:
                print("No return for:", path)
                if self.log:
                    print(self.log_txt)
            else:
                print("Ok for:", path)
            os.chdir(old)
            return result
        except BaseException as e:
            print(e)
            print("Error for:", path)
            if self.log:
                print(self.log_txt)
            os.chdir(old)
            return None

    @property
    def log_txt(self):
        return f"""
        ##### HELP START #######"
        1.Check the necessary files, and software:
        {self.necessary_files}, {self.software}
        2.Check the necessary files from key parameter:
        {self.key_help}
        3.Check the intermediate temporary files are generated in path:
        {self.inter_temp_files}
        "##### HELP END #######"
        """

    @abstractmethod
    def run(self, path: Path, files: List = None) -> Any:  # fill
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""

        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # Sample
        # vasprun = Vasprun(path/"vasprun.xml")
        # ef = vasprun.efermi
        # return {"efermi":ef}

        return {"File": path}

    @abstractmethod
    def batch_after_treatment(self, paths, res_code):  # fill
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        # 合并多个文件到一个csv.

        # Sample
        # data_all = {pi:ri for pi,ri in zip(paths,res_code)}
        # result = pd.DataFrame(data_all).T
        # result.to_csv(self.out_file)
        # print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))

        return pd.DataFrame.from_dict({"File": paths}).T


class BaderStartZero(_BasePathOut):
    """Get bader from paths and return csv file.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["AECCAR0", "AECCAR2", "CHGCAR", "POTCAR", "POSCAR"]
        self.out_file = "bader_all.csv"
        self.software = ["chgsum.pl", "bader"]
        self.inter_temp_files = ["ACF.dat", "CHGCAR_sum"]
        self.key_help = "Make 'LAECHG=.TRUE.' 'LCHARG = .TRUE.'  'IBRION = -1'  in VASP INCAR."

    @staticmethod
    def read(path, store=False, store_name="bader_single.csv"):
        """Run linux cmd and return result, make sure the bader is installed."""
        if (path / "POTCAR").isfile() and (path / "POSCAR").isfile:
            potcar = Potcar.from_file("POTCAR")
            symbols = Poscar.from_file("POSCAR", check_for_POTCAR=False).structure.atomic_numbers

            zval = []
            for i in symbols:
                for j in potcar:
                    if j.atomic_no == i:
                        zval.append(j.ZVAL)
                        break

            zval = np.array(zval)

            with open("ACF.dat", mode="r") as f:
                msg = f.readlines()

            msg = [i for i in msg if ":" not in i]
            msg = [i for i in msg if "--" not in i]
            msg = [i.replace("\n", "") if "\n" in i else i for i in msg]
            msg = [i.split(" ") for i in msg[1:]]
            msg = [[k for k in i if k != ""] for i in msg]
            msg2 = np.array(msg).astype(float)

            msg_append = (msg2[:, 4] - zval).reshape(-1, 1)
            msg_f = np.full(msg2.shape[0], fill_value=str(path)).reshape(-1, 1)
            msg2 = np.concatenate((msg_f, msg2, msg_append), axis=1)
            ta = np.array(["File", "Atom Number", "X", "Y", "Z", "CHARGE", " MIN DIST", " ATOMIC VOL",
                           "Bader Ele Move"])

            result = pd.DataFrame(msg2, columns=ta)
            if store:
                result.to_csv(store_name)
                print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

            return result
        else:
            return None

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # 1.外部软件
        cmds = ("chgsum.pl AECCAR0 AECCAR2", "bader CHGCAR -ref CHGCAR_sum")
        for i in cmds:
            os.system(i)

        return self.read(path, store=self.store_single, store_name="bader_single.csv")

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = []
        col = None
        for res in res_code:
            if isinstance(res, pd.DataFrame):
                col = res.columns
                res = res.values
                data_all.append(res)

        data_all = np.concatenate(data_all, axis=0)
        result = pd.DataFrame(data_all, columns=col)

        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))
        return result


class BaderStartInter(BaderStartZero):
    """
    For some system can't run this BaderStartZero.

    1. Copy follow code to form one ”badertoACF.sh“ file, and 'sh badertoACF.sh':
    ############
    #!/bin/bash
    old_path=$(cd "$(dirname "$0")"; pwd)
    for i in $(cat paths.temp)
    do
    cd $i
    echo $(cd "$(dirname "$0")"; pwd)

    chgsum.pl AECCAR0 AECCAR2
    bader CHGCAR -ref CHGCAR_sum

    cd $old_path
    done
    ########

    2. tar -zcvf data.tar.gz */*/ACF.dat */*/POTCAR */*/POSCAR

    3. move to other system and 'tar -zxvf data.tar.gz'

    4. Run with this class.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["ACF.dat", "POTCAR", "POSCAR"]
        self.out_file = "bader_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        return self.read(path, store=self.store_single, store_name="bader_single.csv")


class BaderStartSingleResult(BaderStartZero):
    """Avoid Double Calculation. Just reproduce the 'results_all' from a 'result_single' files.
    keeping the 'result_single.csv' files exists.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(BaderStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["bader_single.csv"]
        self.out_file = "bader_all.csv"
        self.software = []
        self.key_help = self.__doc__

    def read(self, path, **kwargs):
        pii = path / self.necessary_files[0]
        if pii.isfile():
            result = pd.read_csv(pii)
            return result
        else:
            return None

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        return self.read(path)


class CLICommand:
    """
    批量提取 bader，保存到当前工作文件夹。 查看参数帮助使用 -h。
    在 featurebox 中运行，请使用 featurebox bader ...
    若复制本脚本并单运行，请使用 python bader.py ...

    Notes:
        1.1 前期准备
        需要以下文件：~/bin/bader, ~/bin/chgsum.pl
        并赋予权限:
        chmod u+x ~/bin/chgsum.pl
        chmod u+x ~/bin/bader

        1.2 前期准备
        INCAR文件参数要求：
        LAECHG = .TRUE.
        LCHARG = .TRUE.
        NSW = 0
        IBRION = -1

        2.运行文件要求:
        chgsum.pl, bader,
        AECCAR0，AECCAR2，CHGCAR

    如果在 featurebox 中运行多个案例，请指定路径所在文件:

    Example:

    $ featurebox bader -f /home/sdfa/paths.temp

    如果在 featurebox 中运行单个案例，请指定运算子文件夹:

    Example:

    $ featurebox bader -p /home/parent_dir/***/sample_dir
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=0)

    @staticmethod
    def run(args, parser):
        methods = [BaderStartZero, BaderStartInter, BaderStartSingleResult]
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods[args.job_type](n_jobs=4, store_single=True)
            with open(pf) as f:
                wd = f.readlines()
            bad.transform(wd)
        elif pn.isdir():
            bad = methods[args.job_type](n_jobs=1, store_single=True)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """
    # os.chdir("./data")
    import argparse

    parser = argparse.ArgumentParser(description="Get bader charge. Examples:\n"
                                                 "python this.py -p /home/dir_name , or\n"
                                                 "python this.py -f /home/dir_name/paths.temp")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
    parser.add_argument('-j', '--job_type', type=int, default=0)
    args = parser.parse_args()
    # run
    methods = [BaderStartZero, BaderStartInter, BaderStartSingleResult]
    pf = Path(args.paths_file)
    pn = Path(args.path_name)
    if pf.isfile():
        bad = methods[args.job_type](n_jobs=1, store_single=True)
        with open(pf) as f:
            wd = f.readlines()
        res_all = bad.transform(wd)
    elif pn.isdir():
        bad = methods[args.job_type](n_jobs=1, store_single=True)
        res = bad.convert(pn)
    else:
        raise NotImplementedError("Please set -f or -p parameter.")


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
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster import Icohplist, Cohpcar
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


class COHPStartZero(_BasePathOut):
    """Get d band center from paths and return csv file.
    VASPKIT Version: 1.2.1  or below.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(COHPStartZero, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["WAVECAR", "INCAR", "DOSCAR"]
        self.out_file = "COHP_all.csv"
        self.software = ["lobster"]

    @staticmethod
    def read(path, store=False):
        data_all = {}
        if os.path.isfile("ICOHPLIST.lobster"):
            icohplist = Icohplist(are_coops=False, are_cobis=False, filename="ICOHPLIST.lobster")
            data = np.array([
                icohplist.icohplist["1"]["length"],
                -(icohplist.icohplist["1"]["icohp"][Spin.up]),
                -(icohplist.icohplist["1"]["icohp"][Spin.down])
            ]).ravel()
            data_all.update({path: data})

        else:
            print(f"no data for {path}")
            print(f"Check your ``lobsterout``, make sure enough ``NBANDS`` in INCAR")
            print(f"Check your VASP input and output, make sure enough ``LWAVE = .TRUE.`` in INCAR")
            data_all.update({path: None})

        data_all2 = {}
        if os.path.isfile("COHPCAR.lobster"):
            cohpcar = Cohpcar(are_coops=False, are_cobis=False, filename="COHPCAR.lobster")

            data = np.vstack((cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            # data = np.vstack((cohpcar.energies, cohpcar.cohp_data["average"]["COHP"][Spin.up])).ravel()
            data_all2.update({path: data})
        else:
            print(f"no data for {path}")
            data_all2.update({path: None})

        result_single1 = pd.DataFrame.from_dict(data_all).T
        result_single2 = pd.DataFrame.from_dict(data_all2).T

        if store:
            result_single1.to_csv("ICOHP_single.csv")
            print("'{}' are sored in '{}'".format("ICOHP_single.csv", os.getcwd()))
            result_single2.to_csv("COHP_single.csv")
            print("'{}' are sored in '{}'".format("COHP_single.csv", os.getcwd()))

        return result_single1

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # 1.外部软件
        cmds = ("lobster > look",)
        for i in cmds:
            print("Runing waiting for ")
            os.system(i)

        return self.read(path, store=self.store_single)

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = []
        col = None
        for pi in paths:
            try:
                res = pd.read_csv(pi / "ICOHP_single.csv", index_col=0)
                col = res.columns
                res = res.values
                data_all.append(res)
            except BaseException:
                print(f"No data for {pi}")

        data_all = np.concatenate(data_all, axis=0)
        result = pd.DataFrame(data_all, columns=col, index=paths)

        result.to_csv("ICOHP_all.csv")
        print("'{}' are sored in '{}'".format("ICOHP_all.csv", os.getcwd()))

        data_all2 = []
        col = None
        for pi in paths:
            try:
                res = pd.read_csv(pi / "COHP_single.csv", index_col=0)
                col = res.columns
                res = res.values
                data_all2.append(res)
            except BaseException:
                print(f"No data for {pi}")

        data_all2 = np.concatenate(data_all2, axis=0)
        result = pd.DataFrame(data_all2, columns=col, index=paths)

        result.to_csv("COHP_all.csv")
        print("'{}' are sored in '{}'".format("COHP_all.csv", os.getcwd()))

        return result


class COHPStartInter(COHPStartZero):
    """
    For some system can't run this COHPStartZero.

    1. Copy follow code to form one ”lob.sh“ file, and 'sh lob.sh' (change the atoms couple):

    ############
    #!/bin/bash
    old_path=$(cd "$(dirname "$0")"; pwd)
    for i in $(cat paths.temp)
    do
    cd $i
    echo $(cd "$(dirname "$0")"; pwd)

    echo COHPStartEnergy -10 > lobsterin
    echo COHPEndEnergy 5 >> lobsterin
    echo cohpBetween atom 45 atom 31 >> lobsterin

    lobster > look

    cd $old_path
    done
    ########

    2. tar -zcvf data.tar.gz "ICOHPLIST.lobster", "COHPCAR.lobster"

    3. move to other system and 'tar -zxvf data.tar.gz'

    4. Run with this class.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(COHPStartInter, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["ICOHPLIST.lobster", "COHPCAR.lobster"]
        self.out_file = "ICOHP_all.csv"
        self.software = []

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        return self.read(path, store=self.store_single)


class COHPStartSingleResult(COHPStartInter):
    """Avoid Double Calculation."""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(COHPStartSingleResult, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["COHP_single.csv", "ICOHP_single.csv"]
        self.out_file = "ICOHP_all.csv"
        self.software = []

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
    批量提取 ICOHP，保存到当前工作文件夹。 查看参数帮助使用 -h。

    在 featurebox 中运行，请使用 featurebox cohp ...
    若复制本脚本并单运行，请使用 python cohp ...

    Notes:
        1.1 前期准备
        "WAVECAR", "INCAR", "DOSCAR"

        2.运行文件要求:

    如果在 featurebox 中运行多个案例，请指定路径所在文件:

    Example:

    $ featurebox cohp -f /home/sdfa/paths.temp

    如果在 featurebox 中运行单个案例，请指定运算子文件夹:

    Example:

    $ featurebox cohp -p /home/parent_dir/***/sample_dir
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=1)

    @staticmethod
    def run(args, parser):
        methods = [COHPStartZero, COHPStartInter, COHPStartSingleResult]
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods[args.job_type](n_jobs=4)
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
    """
    import argparse

    # os.chdir("./data2")

    parser = argparse.ArgumentParser(description="Get COHP. Examples：\n"
                                                 "python this.py -p /home/dir_name")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
    parser.add_argument('-j', '--job_type', type=int, default=1)

    args = parser.parse_args()
    # run
    methods = [COHPStartZero, COHPStartInter, COHPStartSingleResult]
    pf = Path(args.paths_file)
    pn = Path(args.path_name)
    if pf.isfile():
        bad = methods[args.job_type](n_jobs=1, store_single=True)
        with open(pf) as f:
            wd = f.readlines()
        bad.transform(wd)
    elif pn.isdir():
        bad = methods[args.job_type](n_jobs=1, store_single=True)
        bad.convert(pn)
    else:
        raise NotImplementedError("Please set -f or -p parameter.")

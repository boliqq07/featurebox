# -*- coding: utf-8 -*-
# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import multiprocessing
import os
import pandas as pd
import pathlib
import warnings

from abc import abstractmethod
from path import Path
from tqdm import tqdm
from typing import Union, Callable, List, Any


class _BasePathOut2:
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

    def convert(self, path2: List[Union[os.PathLike, Path, pathlib.Path, str]]):
        self.check_software()
        path2 = [self._to_path(path) for path in path2]
        path_bool = all([self.check_path_and_file(path) for path in path2])
        if path_bool:
            return self.wrapper(path2)
        else:
            raise FileNotFoundError

    def transform(self, paths: List[List[Union[os.PathLike, Path, pathlib.Path, str]]]):
        self.check_software()
        paths = [[self._to_path(pi) for pi in path] for path in paths]
        paths_bool = [all([self.check_path_and_file(pi) for pi in path]) for path in paths]
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

    def wrapper(self, paths: List[Path]):
        """For exception."""

        try:
            result = self.run(paths)
            if result is None:
                print("No return for:", paths)
                if self.log:
                    print(self.log_txt)
            else:
                print("Ok for:", paths)
            return result
        except BaseException as e:
            print(e)
            print("Error for:", paths)
            if self.log:
                print(self.log_txt)
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
    def run(self, paths: List[Path], files: List = None) -> Any:  # fill
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""

        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)

        # Sample
        # vasprun = Vasprun(path/"vasprun.xml")
        # ef = vasprun.efermi
        # return {"efermi":ef}

        return {"File": paths}

    @abstractmethod
    def batch_after_treatment(self, paths: List[List[Path]], res_code):  # fill
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        # 合并多个文件到一个csv.

        # Sample
        # data_all = {pi:ri for pi,ri in zip(paths,res_code)}
        # result = pd.DataFrame(data_all).T
        # result.to_csv(self.out_file)
        # print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))

        return pd.DataFrame.from_dict({"File": paths}).T


class GeneralDiff(_BasePathOut2):
    """Get data from paths and return csv file."""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False,
                 mod="pymatgen.io.vasp", cmd="Vasprun", necessary_files="vasprun.xml", prop="final_energy"):
        super(GeneralDiff, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        from importlib import import_module
        mod = import_module(mod)
        self.cmd = getattr(mod, cmd)
        self.necessary_files = [necessary_files, ]
        self.prop = prop
        self.out_file = f"{prop}-diff_all.csv"
        self.software = []

    def run(self, paths: List[Path], files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        res = []
        for path in paths:
            try:
                vasprun = self.cmd(path / self.necessary_files[0])
            except BaseException:
                vasprun = self.cmd.from_file(path / self.necessary_files[0])

            data = getattr(vasprun, self.prop)

            if isinstance(data, (float, int)):
                res.append(data)
            else:
                raise NotImplementedError
        data = {f"{self.prop}-diff": res[0] - res[1]}

        if self.store_single:
            result = pd.DataFrame.from_dict({f"{paths[0]}-{paths[1]}": data}).T
            result.to_csv(f"{self.prop}-diff_single.csv")
            print(f"Store {self.prop}-diff_single.csv to {os.getcwd()}.")
            # This file would be covered！！！

        return data

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = {f"{pi[1]}-{pi[0]}": ri for pi, ri in zip(paths, res_code)}
        result = pd.DataFrame(data_all).T
        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))


class CLICommand:
    """
    批量获取性质差。 查看参数帮助使用 -h。
    在 featurebox 中运行，请使用 featurebox diff ...
    若复制本脚本并单运行，请使用 python diff ...

    如果在 featurebox 中运行多个案例，请指定路径所在文件:

    Example:

    $ featurebox diff -f /home/sdfa/paths1.temp /home/sdfa/paths2.temp -nec vasprun.xml -prop final_energy

    如果在 featurebox 中运行单个案例，请指定运算子文件夹:

    Example:

    $ featurebox diff -p /home/sdfa/ /home/sdfa2/ -cmd Vasprun -nec vasprun.xml -prop final_energy
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default=("./data", "./data2"), nargs=2)
        parser.add_argument('-f', '--paths_file', type=str, default=("paths1.temp", "paths2.temp"), nargs=2)
        parser.add_argument('-mod', '--mod', type=str, default='pymatgen.io.vasp')
        parser.add_argument('-cmd', '--cmd', type=str, default='Vasprun')
        parser.add_argument('-nec', '--nec', type=str, default='vasprun.xml')
        parser.add_argument('-prop', '--prop', type=str, default='final_energy')
        # mod = "pymatgen.io.vasp", cmd = "Vasprun", necessary_files = "vasprun.xml", prop = "final_energy"

    @staticmethod
    def run(args, parser):

        pf = [Path(i) for i in args.paths_file]
        pn = [Path(i) for i in args.path_name]
        if pf[0].isfile():
            bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, n_jobs=1)
            with open(pf[0]) as f1:
                wd1 = f1.readlines()
            with open(pf[1]) as f2:
                wd2 = f2.readlines()
            bad.transform([list(i) for i in zip(wd1, wd2)])
        elif pn[0].isdir():
            bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, store_single=False)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:
        $ python this.py -p /home/dir_name /home/dir_name2
        $ python this.py -f /home/dir_name/path.temp1  /home/dir_name/path.temp2  
    """
    import argparse

    parser = argparse.ArgumentParser(description="Get d band centor.Examples:\n"
                                                 "python this.py -p /home/dir_name /home/dir_name2")

    parser.add_argument('-p', '--path_name', type=str, default=("./data", "./data2"), nargs=2)
    parser.add_argument('-f', '--paths_file', type=str, default=("paths1.temp", "paths2.temp"), nargs=2)
    parser.add_argument('-mod', '--mod', type=str, default='pymatgen.io.vasp')
    parser.add_argument('-cmd', '--cmd', type=str, default='Vasprun')
    parser.add_argument('-nec', '--nec', type=str, default='vasprun.xml')
    parser.add_argument('-prop', '--prop', type=str, default='final_energy')

    args = parser.parse_args()
    # run
    pf = [Path(i) for i in args.paths_file]
    pn = [Path(i) for i in args.path_name]
    if pf[0].isfile():
        print(args.paths_file)
        bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, n_jobs=1)
        with open(pf[0]) as f1:
            wd1 = f1.readlines()
        with open(pf[1]) as f2:
            wd2 = f2.readlines()
        bad.transform([list(i) for i in zip(wd1, wd2)])
    elif pn[0].isdir():
        print(args.path_name)
        bad = GeneralDiff(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, store_single=False)
        bad.convert(pn)
    else:
        raise NotImplementedError("Please set -f or -p parameter.")

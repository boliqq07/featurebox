# -*- coding: utf-8 -*-

# @Time  : 2022/8/3 14:00
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import multiprocessing
import os
import pathlib
import warnings
from abc import abstractmethod
from typing import Union, Callable, List, Any

import pandas as pd
import path
from path import Path
from tqdm import tqdm


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


def check_convergence(pt: Union[str, path.Path, os.PathLike, pathlib.Path], msg=None):
    """
    Check final energy.

    检查结构是否收敛。

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path
        msg:(list of str), message.

    Returns:
        res:(tuple), bool and msg list

    """
    if msg is None:
        msg = []
    msg.append("\nCheck Convergence:")
    key_sentence0 = ' reached required accuracy - stopping structural energy minimisation\n'
    key_sentence1 = ' General timing and accounting informations for this job:\n'
    if not isinstance(pt, path.Path):
        pt = path.Path(pt)
    try:
        with open(pt / 'OUTCAR') as c:
            outcar = c.readlines()

        if key_sentence0 not in outcar[-40:] and key_sentence1 not in outcar[-20:]:
            warnings.warn(f"Not converge and not get the final energy.",
                          UnicodeWarning)
            msg.append(f"Not converge and not get the final energy.")
            res = False, msg
        else:
            res = True, msg

    except BaseException:
        warnings.warn(f"Error to read OUTCAR.",
                      UnicodeWarning)
        msg.append(f"Error to read OUTCAR.")

        res = False, msg

    return res


class ConvergeChecker(_BasePathOut):
    """Get d band center from paths and return csv file.
    VASPKIT Version: 1.2.1  or below.
    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False):
        super(ConvergeChecker, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = ["OUTCAR"]
        self.out_file = "Fail_paths.temp"
        self.software = []
        self.key_help = "Check the outcar."

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)
        res_bool=check_convergence(path)[0]
        if res_bool:
            pass
        else:
            if self.store_single:
                with open("Fail_this_paths_single.temp", "w") as f:
                    f.writelines(path)
        return res_bool

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        res_fail = [pathi for pathi,resi in zip(paths,res_code) if resi is False]
        with open(f"{self.out_file}", "w") as f:
            f.writelines("\n".join(res_fail))
        return res_code

class CLICommand:
    """
    批量检查outcar，保存到当前工作文件夹。 查看参数帮助使用 -h。

    在 featurebox 中运行，请使用 featurebox converge ...
    若复制本脚本并单运行，请使用 python converge ...

    如果在 featurebox 中运行多个案例，请指定路径所在文件:

    Example:

    $ featurebox converge -f /home/sdfa/paths.temp

    如果在 featurebox 中运行单个案例，请指定运算子文件夹:

    Example:

    $ featurebox converge -p /home/parent_dir/***/sample_dir
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-j', '--job_type', type=int, default=0)

    @staticmethod
    def run(args, parser):
        methods = [ConvergeChecker]
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods[args.job_type](n_jobs=1, store_single=False)
            with open(pf) as f:
                wd = f.readlines()
            bad.transform(wd)
        elif pn.isdir():
            bad = methods[args.job_type](n_jobs=1, store_single=False)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:

        $ python this.py -p /home/dir_name
    """
    import argparse

    parser = argparse.ArgumentParser(description="Get converge failed path. Example:\n"
                                                 "python this.py -p /home/dir_name")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
    parser.add_argument('-j', '--job_type', type=int, default=0)

    args = parser.parse_args()
    # run
    methods = [ConvergeChecker, ]
    pf = Path(args.paths_file)
    pn = Path(args.path_name)
    if pf.isfile():
        bad = methods[args.job_type](n_jobs=1, store_single=False)
        with open(pf) as f:
            wd = f.readlines()
        bad.transform(wd)
    elif pn.isdir():
        bad = methods[args.job_type](n_jobs=1, store_single=False)
        bad.convert(pn)
    else:
        raise NotImplementedError("Please set -f or -p parameter.")

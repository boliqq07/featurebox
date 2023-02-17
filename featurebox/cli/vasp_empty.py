# -*- coding: utf-8 -*-

# @Time  : 2022/8/3 14:00
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import os
import warnings
from typing import List

from path import Path

from featurebox.cli._basepathout import _BasePathOut


def check_empty(pt, msg=None):
    """
    Check the file is empty or not .

    Args:
        pt: (str, path.Path, os.PathLike,pathlib.Path), path
        msg:(list of str), message.

    Returns:
        res:(tuple), bool and msg list

    """
    if msg is None:
        msg = []
    msg.append("\nCheck Empty:")

    if not isinstance(pt, Path):
        pt = Path(pt)
    try:
        if pt.isfile():
            if os.path.getsize(pt):
                res = True, msg
            else:
                res = False, msg
        else:
            res = False, msg

    except BaseException as e:
        print(pt)
        warnings.warn(f"Error to read CONTCAR.",
                      UnicodeWarning)
        msg.append(f"Error to read CONTCAR.")
        print(e)

        res = False, msg

    return res


class EmptyChecker(_BasePathOut):
    """Check the file is empty or not"""

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False, fn="CONTCAR"):
        super(EmptyChecker, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        self.necessary_files = []
        self.out_file = "Fail_paths.temp"
        self.software = []
        self.key_help = "Check the CONTCAR is empty or not."
        self.extract = None
        self.fn = fn

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        # 可以更简单的直接写命令，而不用此处的文件名 (files), 但是需要保证后续出现的文件、软件与 necessary_file, software 一致
        # 函数强制返回除去None的对象，用于后续检查!!! (若返回None,则认为run转换错误)
        path = path / self.fn

        res_bool = check_empty(path)[0]

        if res_bool:
            pass
        else:
            if self.store_single:
                with open(path / "Fail_this_paths_single.temp", "w") as f:
                    f.writelines(path)
        return res_bool

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        res_fail = [pathi for pathi, resi in zip(paths, res_code) if resi is False]
        with open(f"{self.out_file}", "w") as f:
            f.writelines("\n".join(res_fail))
        return res_code


class _CLICommand:
    """
    批量检查CONTCAR。 查看参数帮助使用 -h。

    补充:

        在 featurebox 中运行，请使用 featurebox empty ...

        若复制本脚本并单运行，请使用 python {this}.py ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox converge -f /home/sdfa/paths.temp

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox converge -p /home/parent_dir/***/sample_dir

    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-n', '--check_file_name', type=str, default='CONTCAR')

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        methods = EmptyChecker
        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = methods(n_jobs=4, store_single=False, fn=args.check_file_name)
            with open(pf) as f:
                wd = f.readlines()
            assert len(wd) > 0, f"No path in file {pf}"
            bad.transform(wd)
        elif pn.isdir():
            bad = methods(n_jobs=1, store_single=False, fn=args.check_file_name)
            bad.convert(pn)
        else:
            raise NotImplementedError("Please set -f or -p parameter.")


if __name__ == '__main__':
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """
    import argparse

    parser = argparse.ArgumentParser(description=f"Get data by {__file__}. Examples:\n"
                                                 "python this.py -p /home/dir_name , or\n"
                                                 "python this.py -f /home/dir_name/paths.temp")
    _CLICommand.add_arguments(parser=parser)
    args = _CLICommand.parse_args(parser=parser)
    _CLICommand.run(args=args, parser=parser)

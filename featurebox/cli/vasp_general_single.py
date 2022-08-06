# -*- coding: utf-8 -*-

# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import os
from typing import List

import numpy as np
import pandas as pd
from path import Path

from featurebox.cli._basepathout import _BasePathOut


class General(_BasePathOut):
    """Get data from paths and return csv file.

    Notes::

        mod="pymatgen.io.vasp"         # Module to get class.
        cmd="Vasprun"                  # class to get object.
        necessary_files="vasprun.xml"  # class input file.
        prop="final_energy"            # class.property name.

    """

    def __init__(self, n_jobs: int = 1, tq: bool = True, store_single=False,
                 mod="pymatgen.io.vasp", cmd="Vasprun", necessary_files="vasprun.xml", prop="final_energy"):
        super(General, self).__init__(n_jobs=n_jobs, tq=tq, store_single=store_single)
        from importlib import import_module
        mod = import_module(mod)
        self.cmd = getattr(mod, cmd)
        self.necessary_files = [necessary_files, ]
        self.prop = prop
        self.out_file = "general_all.csv"
        self.software = []

    def run(self, path: Path, files: List = None):
        """3.Run with software and necessary file and get data.
        (1) Return result in code, or (2) Both return file to each path and in code."""
        try:
            vasprun = self.cmd(path / self.necessary_files[0])
        except BaseException:
            vasprun = self.cmd.from_file(path / self.necessary_files[0])

        data = getattr(vasprun, self.prop)
        if isinstance(data, (tuple, list)):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            if data.shape == 1:
                data = {f"{self.prop}-{n}": i for n, i in enumerate(data)}
            elif data.shape == 0:
                data = {self.prop: data[0]}
            else:
                raise NotImplementedError
        elif isinstance(data, dict):
            data = data
        elif isinstance(data, (float, int)):
            data = {self.prop: data}
        else:
            raise NotImplementedError

        if self.store_single:
            result = pd.DataFrame.from_dict(data).T
            result.to_csv("general_single.csv")

        return data

    def batch_after_treatment(self, paths, res_code):
        """4. Organize batch of data in tabular form, return one or more csv file. (force!!!)."""
        data_all = {pi: ri for pi, ri in zip(paths, res_code)}
        result = pd.DataFrame(data_all).T
        result.to_csv(self.out_file)
        print("'{}' are sored in '{}'".format(self.out_file, os.getcwd()))


class _CLICommand:
    """
    批量获取性质（默认vasp.xml能量）。 查看参数帮助使用 -h。

    补充:

        在 featurebox 中运行，请使用 featurebox general ...

        若复制本脚本并单运行，请使用 python general ...

        如果在 featurebox 中运行多个案例，请指定路径所在文件:

        $ featurebox general -f /home/sdfa/paths.temp -cmd Vasprun -nec vasprun.xml -prop final_energy

        如果在 featurebox 中运行单个案例，请指定运算子文件夹:

        $ featurebox general -f /home/sdfa/paths.temp -cmd Vasprun -nec vasprun.xml -prop final_energy
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-f', '--paths_file', type=str, default='paths.temp')
        parser.add_argument('-mod', '--mod', type=str, default='pymatgen.io.vasp')
        parser.add_argument('-cmd', '--cmd', type=str, default='Vasprun')
        parser.add_argument('-nec', '--nec', type=str, default='vasprun.xml')
        parser.add_argument('-prop', '--prop', type=str, default='final_energy')

        # mod = "pymatgen.io.vasp", cmd = "Vasprun", necessary_files = "vasprun.xml", prop = "final_energy"

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):

        pf = Path(args.paths_file)
        pn = Path(args.path_name)
        if pf.isfile():
            bad = General(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, n_jobs=4)
            with open(pf) as f:
                wd = f.readlines()
            assert len(wd) > 0, f"No path in file {pf}"
            bad.transform(wd)
        elif pn.isdir():
            bad = General(mod=args.mod, cmd=args.cmd, necessary_files=args.nec, prop=args.prop, store_single=True)
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

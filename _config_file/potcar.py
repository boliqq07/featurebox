# -*- coding: utf-8 -*-

# @Time     : 2021/8/24 13:57
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

#####################################
import os
import warnings

try:
    from pymatgen.core import SETTINGS
except (ImportError,TypeError):
    SETTINGS =None


def check_PMG_VASP_PSP_DIR():
    d = SETTINGS.get("PMG_VASP_PSP_DIR")
    if d is None:
        warnings.warn(
            "No POTCAR found. Please set the PMG_VASP_PSP_DIR environment in .pmgrc.yaml by `set_PMG_VASP_PSP_DIR function` or manually",
            UserWarning)
        return False
    else:
        return True


def set_PMG_VASP_PSP_DIR(abspath):
    """
    一. abspath 文件夹产生方式为：

    # 初始potcar, 位置为 EXTRACTED_VASP_POTCAR （初始poscar数据源在华为盘，百度盘，240）
    # 1.`pmg config - p <EXTRACTED_VASP_POTCAR>  ../../poscar_pmg  ` 产生 poscar_pmg 文件夹。
    # 2.`pmg config --add PMG_VASP_PSP_DIR <MY_PSP> ` 设置 poscar_pmg 文件夹。

    二. 如果直接有 poscar_pmg 文件夹，可使用此函数设置 poscar_pmg 路径，也可以直接用第二步。
    """
    if SETTINGS is not None:
        d = SETTINGS.get("PMG_VASP_PSP_DIR")
    else:
        d= None

    SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".pmgrc.yaml")
    if d is None:
        if os.path.isfile(SETTINGS_FILE):
            with open(SETTINGS_FILE, "a+") as f:
                f.write("\nPMG_VASP_PSP_DIR: {}".format(abspath))
        else:
            with open(SETTINGS_FILE, "w") as f:
                f.write("\nPMG_VASP_PSP_DIR: {}".format(abspath))

    else:
        with open(SETTINGS_FILE, "r") as f:
            lines = f.readlines()
        lines = [i for i in lines if "VASP_PSP_DIR" not in i]

        with open(SETTINGS_FILE, "w+") as f:
            f.writelines(lines)
            f.write("\nPMG_VASP_PSP_DIR: {}".format(abspath))

        print("PMG_VASP_PSP_DIR is refreshed to `{}`".format(abspath))


# from pymatgen.io.vasp import Potcar, Poscar
# class Potcar2(Potcar):
#
#     def __init__(self, symbols=None, functional=None, sym_potcar_map=None):
#         assert check_PMG_VASP_PSP_DIR()
#         super().__init__(symbols=symbols, functional=functional, sym_potcar_map=sym_potcar_map)
#
#     @classmethod
#     def from_poscar(cls, poscar: Poscar, functional=None, sym_potcar_map=None):
#         symbol = poscar.site_symbols
#         return cls(symbols=symbol, functional=functional, sym_potcar_map=sym_potcar_map)

if __name__ == '__main__':

    set_PMG_VASP_PSP_DIR("/data/home/suyj/wcx/potcars_pmg/")

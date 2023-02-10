#!/bin/bash
import os
import re


def get_manager():
    return _get_manager()[0]


def _get_manager():

    res2 = os.popen("whereis jsub").readlines()[0]
    res22 = re.search("(/\S+)+", res2)
    if res22:
        return "jsub", res22.group()

    res1 = os.popen("whereis sbatch").readlines()[0]
    res11 = re.search("(/\S+)+", res1)
    if res11:
        return "sbatch", res11.group()

    res3 = os.popen("whereis qsub").readlines()[0]
    res33 = re.search("(/\S+)+", res3)
    if res33:
        return "qsub", res33.group()
    else:
        raise NotImplementedError("No sbatch, jsub, qsub find.")


def set_bachrc(path="{home}/history_jobs", log_paths_file="paths.temp"):
    if "{home}" in path:
        home = os.path.expandvars('$HOME')
        path = path.replace("{home}", home)

    if os.path.isfile(f"{path}/.boxrc"):
        pass
    else:
        a, b = _get_manager()
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(f"{path}/.boxrc", "w+") as f:
            f.write("MAX_SIZE=1000\n"
                    f"CMD_TYPE={a}\n"
                    f"CMD_EXE={b}\n")
        if not os.path.isfile(f"{path}/{log_paths_file}"):
            with open(f"{path}/{log_paths_file}", "w") as f:
                f.write("")

        text = f'alias {a}="pwd >> {path}/{log_paths_file} && {b}"\n'

        home = os.path.expandvars('$HOME')
        with open(f"{home}/.bashrc", "r") as f:
            wods = f.readlines()
        if text not in wods:
            with open(f"{home}/.bashrc", "a+") as f:
                wods.append(text)
                f.writelines(wods)


def reform_log_path(max_size=None, path="{home}/history_jobs", log_paths_file="paths.temp"):
    """cut the log_paths_file, and left the latest max_size paths."""
    if "{home}" in path:
        home = os.path.expandvars('$HOME')
        path = path.replace("{home}", home)

    if max_size is None:
        max_size = 1000

        ###################### read MAX_SIZE form file, we now not used.
        # if os.path.isfile(f"{path}/.boxrc"):
        #     with open(f"{path}/.boxrc") as f:
        #         wd = f.readlines()
        #         for wi in wd:
        #             if "MAX_SIZE" in wi:
        #                 size = wi.split("=")[1]
        #                 size = size.replace(" ", "").rstrip()
        #                 max_size = int(size)
        #                 break
        ###################

    assert max_size > 100

    with open(f"{path}/{log_paths_file}", "r") as f:
        wods = f.readlines()
        if len(wods) > max_size:
            cut = True
        else:
            cut = False
    if cut:
        with open(f"{path}/{log_paths_file}", "w+") as f:
            wods = wods[-(max_size - 100):]
            f.writelines(wods)

    return wods


if __name__ == "__main__":
    ls = set_bachrc()

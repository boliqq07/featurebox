#!/bin/bash
import json
import os.path
import time
from typing import Union, List, Sequence

import pandas as pd

from featurebox.pbs.pbs_conf import set_bachrc, reform_log_path


class JobManager:

    def __init__(self, manager: str = "torque",):

        set_bachrc(path="{home}/history_jobs", log_paths_file="paths.temp")
        reform_log_path(max_size=1000, path="{home}/history_jobs", log_paths_file="paths.temp")

        if manager in ("squeue", "slurm", "sbatch"):
            from featurebox.pbs import misc_slurm
            self.funcs = misc_slurm
        elif manager in ("pbs", "torque", "qsub", "qstat"):
            from featurebox.pbs import misc_torque
            self.funcs = misc_torque
        elif manager in ("jjobs", "unischeduler", "jctrl", "jsub"):
            try:
                from featurebox.pbs import misc_unischeduler
            except ImportError:
                import misc_unischeduler
            self.funcs = misc_unischeduler
        else:
            raise NotImplementedError

        self.msg = {}
        tm_year, tm_mon, tm_mday, _, _, _, _, _, _ = time.localtime()
        self.time_tup = tm_year, tm_mon, tm_mday
        self.home = os.path.expandvars('$HOME')
        self.deleted_msg = {}

    def change_manager(self, manager: str = "torque"):
        self.__init__(manager=manager)

    def get_job_msg(self, jobid: Union[str, List[str]] = None):
        self.msg = self.funcs.job_status(jobid)
        return self.msg

    def load_job_msg(self, file=None, path="{home}/history_jobs"):
        path = path.replace("{home}", self.home)

        if file is None:
            file = f"{path}/{self.time_tup[0]}-{self.time_tup[1]}-{self.time_tup[2]}/.json"
        if os.path.isfile(file):
            with open(file, "r") as fp:
                msg = json.load(fp)
                msg.update(self.msg)
                self.msg = msg

    def dump_job_msg(self, file=None, path="{home}/history_jobs"):

        path = path.replace("{home}", self.home)
        if file is None:
            file = f"{path}/{self.time_tup[0]}-{self.time_tup[1]}-{self.time_tup[2]}.json"

        if os.path.isfile(file):
            with open(file, "r") as fp:
                old_msg = json.load(fp)
                old_msg.update(self.msg)
        else:
            old_msg = self.msg

        with open(file, "w") as fp:
            json.dump(old_msg, fp)

    def sparse(self):
        res = {}
        for k, v in self.msg.items():
            ze = {
                "procs": v["procs"],
                "jobstatus": v["jobstatus"],
                "starttime": time.ctime(v["starttime"]),
                "elapsedtime": time.ctime(v["elapsedtime"]),
                "completiontime": time.ctime(v["completiontime"]),
                "work_dir": v["work_dir"]}
            res.update({k: ze})
        return res

    def print_pd(self):
        res = self.sparse()
        res = pd.DataFrame.from_dict(res).T
        return res

    def submit_from_path(self, path: Union[str, list, tuple], file: str = "*.run") -> Sequence:
        if isinstance(path, (tuple, list)):
            jobids = [self.funcs.submit_from_path(pathi, file) for pathi in path]
        elif isinstance(path, str):
            jobids = [self.funcs.submit_from_path(path, file)]
        else:
            raise NotImplementedError
            # return []

        jobids = [i for i in jobids if i is not None]

        if len(jobids) > 0:
            msg = self.funcs.job_status(jobids)
            self.msg.update(msg)
        else:
            msg = self.funcs.job_status()
            self.msg.update(msg)
        return jobids

    def re_submit_from_path(self, path=None, file="*.run", old_id: Union[str, list] = None):
        if path is not None:
            return self.submit_from_path(path, file)
        else:
            if isinstance(old_id, str):
                old_id = [old_id, ]

            new_id = []

            for old_idi in old_id:
                if old_idi is not None:
                    if old_idi in self.msg:
                        path = self.msg[old_idi]["work_dir"]
                        self.delete(old_idi)
                        ni = self.submit_from_path(path=path, file=file)
                        new_id.extend(ni)
                    elif old_idi in self.deleted_msg:
                        path = self.deleted_msg[old_idi]["work_dir"]
                        ni = self.submit_from_path(path=path, file=file)
                        new_id.extend(ni)
                    else:
                        print(f"Not find the old jobs, please using job 'path'.")
            return new_id

    def clear(self):
        sg = [i for i in self.msg.keys()]
        self.funcs.clear(sg)
        self.deleted_msg.update(self.msg)
        self.msg.clear()
        return sg

    def delete(self, jobid: Union[str, list]):
        if isinstance(jobid, str):
            self.funcs.delete(jobid)
            v = self.msg.pop(jobid)
            self.deleted_msg.update({jobid: v})
        else:
            self.funcs.clear(jobid)
            v = [self.msg.pop(i) for i in jobid]
            [self.deleted_msg.update({ji: vi}) for ji, vi in zip(jobid, v)]
        return jobid

    def hold(self, jobid: Union[str, list]):
        if isinstance(jobid, (list, tuple)):
            return [self.hold(i) for i in jobid]
        else:
            self.funcs.hold(jobid)
            msg = self.funcs.job_status(jobid)
            self.msg.update(msg)
        return jobid

    def release(self, jobid: Union[str, list]):
        if isinstance(jobid, (list, tuple)):
            return [self.release(i) for i in jobid]
        else:
            self.funcs.release(jobid)
            msg = self.funcs.job_status(jobid)
            self.msg.update(msg)
        return jobid

    def job_id(self):
        return self.funcs.job_id()

    def job_dir(self, jobid=None):
        return self.funcs.job_rundir(jobid)


if __name__ == "__main__":
    jm1 = JobManager()
    pathh = ["/home/wcx/data/code_test/Ta2CO2/Zn/pure",
             "/home/wcx/data/code_test/V2CO2/Zn",
             "/home/wcx/data/code_test/V2CO2/nodoping",
             "/home/wcx/data/code_test/Zr2CO2/Zn/pure_static"]

    jm1.submit_from_path(pathh, file="p*.run")

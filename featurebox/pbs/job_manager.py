import json
import os.path
import time
from typing import Union, List

import pandas as pd

import misc_slurm
import misc_torque
import misc_unischeduler


class JobManager:

    def __init__(self, manager: str = "torque"):
        if manager in ("pbs", "torque", "qsub", "qstat"):
            self.funcs = misc_torque
        elif manager in ("squeue", "slurm", "sbatch"):
            self.funcs = misc_slurm
        elif manager in ("jjobs", "unischeduler", "jctrl", "jsub"):
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
        res = pd.from_dict(res).T
        return res

    def submit_file(self, path:Union[str,list,tuple], file:str="*.run"):
        if isinstance(path, (tuple, list)):
            jobids = [self.funcs.submit_file(pathi, file) for pathi in path]
        else:
            jobids = [self.funcs.submit_file(path, file)]

        jobids = [i for i in jobids if i is not None]

        if len(jobids) > 0:
            msg = self.funcs.job_status(jobids)
            self.msg.update(msg)
        else:
            msg = self.funcs.job_status(jobids)
            self.msg.update(msg)

    def re_submit_file(self, path=None, file="*.run", old_ids: str = None,):
        if path and file:
            return self.submit_file(path, file)

        if isinstance(old_ids, str):
            old_ids = [old_ids, ]

        for old_id in old_ids:
            if old_id is not None:
                if old_id in self.msg:
                    path = self.msg[old_id]["work_dir"]
                    self.delete(old_id)
                    self.submit_file(path, file)
                elif old_id in self.deleted_msg:
                    path = self.deleted_msg[old_id]["work_dir"]
                    self.submit_file(path, file)
                else:
                    print(f"Not find the old jobs, please using job 'path'.")

    def clear(self):
        sg = [i for i in self.msg.keys()]
        self.funcs.clear(sg)
        self.deleted_msg.update(self.msg)
        self.msg.clear()

    def delete(self, jobid):
        self.funcs.delete(jobid)
        v = self.msg.pop(jobid)
        self.deleted_msg.update({jobid:v})

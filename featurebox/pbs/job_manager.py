import os.path
from typing import Union, List
import json
import misc_slurm
import misc_torque
import misc_unischeduler
import time

class JobManager:

    def __init__(self, manager: str="torque"):
        if manager in ("pbs", "torque", "qsub", "qstat"):
            self.funcs = misc_torque
        elif manager in ("squeue", "slurm", "sbatch"):
            self.funcs = misc_slurm
        elif manager in ("jjobs", "unischeduler", "jctrl", "jsub"):
            self.funcs = misc_unischeduler
        else:
            raise NotImplementedError

        self.msg = {}
        tm_year, tm_mon, tm_mday,_, _, _, _, _, _ =  time.localtime()
        self.time_tup = tm_year, tm_mon, tm_mday

    def change_manager(self, manager:str="torque"):
        self.__init__(manager=manager)

    def get_job_msg(self, jobid:Union[str,List[str]]=None):
        self.msg = self.funcs.job_status(jobid)
        return self.msg

    def load_job_msg(self, file=None, path="~/history_jobs"):
        if file is None:
            file = f"{path}/{self.time_tup[0]}-{self.time_tup[1]}-{self.time_tup[2]}/.json"
        if os.path.isfile(file):
            with open(file,"r") as fp:
                msg = json.load(fp)
                msg.update(self.msg)
                self.msg = msg

    def dump_job_msg(self, file=None,path="~/history_jobs"):
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









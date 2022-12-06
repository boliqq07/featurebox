#!/bin/bash
""" Misc functions for interfacing between torque and the pbs module """

import datetime
import os
import re
import time
from typing import Union

from featurebox.pbs.misc import getlogin, seconds, run_popen, shell_to_re_compile_pattern_lru


def job_id(username=getlogin()):
    if username is not None:
        qopt = ["qselect", "-u", username]
    else:
        qopt = "qselect"
    qsout = run_popen(qopt, first=False, join=False)
    if qsout is None or len(qsout) == 0:
        return []
    else:
        jobid = []
        for line in qsout:
            jobid += [line.rstrip("\n")]
        return jobid


def _qstat(jobid=None, full=False, username=getlogin()) -> str:
    opt = ["qstat"]

    if username is not None and jobid is None:
        jobid = job_id(username=getlogin())
        opt += ["-u", username]

    if full:
        opt += ["-f"]

    if jobid is not None:
        if isinstance(jobid, str):
            jobid = [jobid]
        elif isinstance(jobid, list):
            pass
        else:
            raise ValueError("jobid must be str or list of str")
        opt += jobid

    sout = run_popen(opt, first=False, join=True)

    return sout


def job_rundir(jobid=None):
    """Return the directory job "id" was run in using qstat.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if jobid is None:
        res = job_status(jobid=jobid)
        return {k: v["work_dir"] for k, v in res.items()}

    if isinstance(jobid, (list, tuple)):
        for i in jobid:
            stdout = _qstat(jobid=i, full=True)
            if stdout is not None:
                match = re.search("init_work_dir = (.*)", stdout)
                if match is not None:
                    rundir[i] = match.group(1)
    else:
        stdout = _qstat(jobid=jobid, full=True)
        if stdout is not None:
            match = re.search("init_work_dir = (.*)", stdout)
            if match is not None:
                res = match.group(1)
                rundir[jobid] = res.replace("\n", "")

    return rundir


def job_status(jobid=None):
    """Return job status using qstat

       Returns a dict of dict, with jobid as key in outer dict.
       Inner dict contains:
       "name", "nodes", "procs", "walltime",
       "jobstatus": status ("Q","C","R", etc.)
       "qstatstr": result of qstat -f jobid, None if not found
       "elapsedtime": None if not started, else seconds as int
       "starttime": None if not started, else seconds since epoch as int
       "completiontime": None if not completed, else seconds since epoch as int

       *This should be edited to return job_status_dict()'s*
    """
    status = dict()

    stdout = _qstat(jobid=jobid, full=True)

    if stdout is None:
        return {}

    # sout = stdout.splitlines()
    sout = stdout.split("\n\n")

    for line in sout:

        m = re.search(r"Job Id:\s*(.*)", line)
        if m:
            jobstatus = {"jobid": m.group(1), "nodes": None, "procs": None, "walltime": None, "qstatstr": line,
                         "elapsedtime": None, "starttime": None, "completiontime": None, "jobstatus": None,
                         'work_dir': None, 'subtime': None}

            m2 = re.search(r"\s*Job_Name\s*=\s*(.*)\s", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            m3 = re.search(r"\s*Resource_List\.nodes\s*=\s*(.*):ppn=(.*)\s", line)
            if m3:
                jobstatus["nodes"] = m3.group(1)
                jobstatus["procs"] = int(m3.group(1)) * int(m3.group(2))

            m4 = re.search(r"\s*Resource_List\.walltime\s*=\s*(.*)\s", line)
            if m4:
                jobstatus["walltime"] = int(seconds(m4.group(1)))

            m5 = re.search(r"\s*start_time\s*=\s*(.*)\s", line)
            if m5:
                jobstatus["starttime"] = int(time.mktime(datetime.datetime.strptime(
                    m5.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))
            else:
                jobstatus["starttime"] = None

            m6 = re.search(r"\s*job_state\s*=\s*(.*)\s", line)
            if m6:
                jobstatus["jobstatus"] = m6.group(1)
            else:
                jobstatus["jobstatus"] = None

            if jobstatus["jobstatus"] == "R" and jobstatus["starttime"] is not None:
                jobstatus["elapsedtime"] = int(time.time()) - jobstatus[
                    "starttime"]
            else:
                jobstatus["elapsedtime"] = None

            m7 = re.search(r"\s*comp_time\s*=\s*(.*)\s", line)
            if m7:
                jobstatus["completiontime"] = int(time.mktime(datetime.datetime.strptime(
                    m7.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            m8 = re.search("init_work_dir = (.*)", line)
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            m9 = re.search(r"\s*ctime\s*=\s*(.*)\s", line)
            if m9:
                jobstatus["subtime"] = int(time.mktime(datetime.datetime.strptime(
                    m9.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            status[jobstatus["jobid"]] = jobstatus

    return status


def submit_from_path(path: str, file: str):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """
    pt = os.getcwd()
    os.chdir(path)

    if any([i in file for i in ["*", "!", "?", "["]]):
        fs = os.listdir()
        fss = "\n".join(fs)

        smatch = shell_to_re_compile_pattern_lru(file, trans=True)
        res = smatch.findall(fss)
        res = [i for i in res if i in fs]
        assert len(res) == 1, f"There are 1+ file/No file {res} with patten {file}, " \
                              f"using more strict/relax condition."
        file = res[0]

    res = run_popen(["qsub", file])

    os.chdir(pt)

    jobid = res.replace("\n", "")
    return jobid


def delete(jobid: Union[list, str]):
    """qdel a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(["qdel", jobid])
    else:
        res = run_popen(f"qdel " + " ".join(jobid))
    return jobid


def clear(jobids: list = None):
    """qdel a PBS job."""
    if jobids is None:
        res = run_popen("qdel all")
    else:
        res = run_popen(f"qdel " + " ".join(jobids))
    return jobids


def hold(jobid: Union[list, str]):
    """qhold a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(["qhold", jobid])
    else:
        res = run_popen(f"qhold " + " ".join(jobid))
    return jobid


def release(jobid: Union[list, str]):
    """qrls a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(["qrls", jobid])
    else:
        res = run_popen(f"qrls " + " ".join(jobid))
    return jobid


if __name__ == "__main__":
    # res21 = find_executable("qsub")
    res2 = job_status()

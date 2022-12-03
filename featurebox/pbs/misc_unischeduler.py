""" Misc functions for interfacing between torque and the pbs module """

import subprocess
import os
import re
import datetime
import time
import sys
from misc import getversion, getlogin, seconds, PBSError, run_cmd

tl = time.localtime()
year = tl.tm_year


def job_id(username=getlogin()):

    if username is not None:

        qsout = run_cmd(["jjobs", "-u", username])

        lines = qsout.split("\n")

        if len(lines) >= 2:

            jobid = []
            for line in lines[1:]:
                ids = line.split(" ")[0]
                if ids != "":
                    jobid.append(ids)
            return jobid
        else:
            return []


def _jjobs(jobid=None, username=getlogin(), full=False, version=getversion()):
    """Return the stdout of qstat minus the header lines.

       By default, 'username' is set to the current user. the '-l' option

       'id' is a string or list of strings of job ids

       Returns the text of jjobs, minus the header lines
    """

    # -u and -f contradict in earlier versions of Torque
    if full and username is not None and jobid is None:
        # First get all jobs by the user
        jobid = job_id(username=getlogin())

    opt = ["jjobs"]
    # If there are jobid(s), you don't need a username
    if username is not None and jobid is None:
        opt += ["-u", username]
    # But if there are jobid(s) and a username, you need -a to get full output
    elif username is not None and jobid is not None and not full:
        opt += ["-a"]
    # By this point we're guaranteed torque ver >= 5.0, so -u and -f are safe together
    if full:
        opt += ["-l"]
    if jobid is not None:
        if isinstance(jobid, str):
            jobid = [jobid]
        elif isinstance(jobid, list):
            pass
        else:
            print("Error in pbs.misc.jjobs(). type(jobid):", type(jobid))
            sys.exit()
        opt += jobid

    sout = run_cmd(opt)

    # return the remaining text
    return sout

def job_rundir(jobid):
    """Return the directory job "id" was run in using qstat.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if isinstance(jobid, (list,tuple)):
        for i in jobid:
            stdout = _jjobs(jobid=i, full=True)
            match = re.search("CWD <(.*)>, Output", stdout)
            res = match.group(1)
            rundir[i] = res.replace("\n", "")
    else:
        stdout = _jjobs(jobid=jobid, full=True)
        match = re.search("CWD <(.*)>, Output", stdout)
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

    # stdout = _jjobs(jobid=jobid, full=True)

    stdout = """

Job <106823>, Job Name <CX_W>, User <wangchangxin>, Project <default>, Status <
                     PEND>, Queue <cpu>, Application <default>, APS Priority <5
                     000>, Command <#!/bin/sh;#JSUB -J CX_W;#JSUB -n 64;#JSUB -
                     R span[ptile=64];#JSUB -q cpu;#JSUB -o out.%J;#JSUB -e err
                     .%J;source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64
                     ;source ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel
                     64;source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64;
                     export PATH=~/app/vasp.5.4.4.fix/bin/:$PATH;ulimit -s 5120
                     000;source /beegfs/jhinno/unischeduler/conf/unisched;#####
                     ###################################################;#   $J
                     H_NCPU:         Number of CPU cores              #;#   $JH
                     _HOSTFILE:     List of computer hostfiles       #;########
                     ################################################;mpirun -n
                     p $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log>
Wed Nov 30 21:36:23: Submitted from host <mu01>, CWD </beegfs/home/wangchangxin
                     /yang/capacity/test-cpu>, Output File <out.106823>, Error
                     File <err.106823>, 64 Processors Requested, Requested Reso
                     urce <span[ptile=64]>;
 ORDER: 79
 PENDING REASONS:
 Load information unavailable: 4 hosts;
 Job slot limit reached: 26 hosts;

 SCHEDULING PARAMETERS:
              r15s     r1m     r5m    r15m      ut      pg      io      ls
 LoadSched       -       -       -       -       -       -       -       -
 LoadStop        -       -       -       -       -       -       -       -

                it     tmp    swap     mem
 LoadSched       -       -       -       -
 LoadStop        -       -       -       -

-------------------------------------------------------------------------------

Job <106979>, Job Name <skk>, User <wangchangxin>, Project <default>, Status <P
                     END>, Queue <cpu>, Application <default>, APS Priority <50
                     00>, Command <#!/bin/sh;#JSUB -J skk;#JSUB -n 64;#JSUB -R
                     span[ptile=64];#JSUB -q cpu;#JSUB -o out.%J;#JSUB -e err.%
                     J;source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64;s
                     ource ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
                     ;source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64;ex
                     port PATH=~/app/vasp.5.4.4.fix/bin/:$PATH;ulimit -s 512000
                     0;source /beegfs/jhinno/unischeduler/conf/unisched;#######
                     #################################################;#   $JH_
                     NCPU:         Number of CPU cores              #;#   $JH_H
                     OSTFILE:     List of computer hostfiles       #;##########
                     ##############################################;mpirun -np
                     $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log>
Sat Dec 03 19:26:52: Submitted from host <mu01>, CWD </beegfs/home/wangchangxin
                     /data/paper2_add/Ti/H_add_relax/S0/00>, Output File <out.1
                     06971>, Error File <err.106971>, 64 Processors Requested,
                     Requested Resource <span[ptile=64]>;
 ORDER: 172
 PENDING REASONS:
 Load information unavailable: 4 hosts;
 Job slot limit reached: 26 hosts;

 SCHEDULING PARAMETERS:
              r15s     r1m     r5m    r15m      ut      pg      io      ls
 LoadSched       -       -       -       -       -       -       -       -
 LoadStop        -       -       -       -       -       -       -       -

                it     tmp    swap     mem
 LoadSched       -       -       -       -
 LoadStop        -       -       -       -


-------------------------------------------------------------------------------

Job <106971>, Job Name <skk>, User <wangchangxin>, Project <default>, Status <E
                     XIT>, Queue <cpu>, Application <default>, APS Priority <50
                     00>, Command <#!/bin/sh;#JSUB -J skk;#JSUB -n 64;#JSUB -R
                     span[ptile=64];#JSUB -q cpu;#JSUB -o out.%J;#JSUB -e err.%
                     J;source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64;s
                     ource ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
                     ;source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64;ex
                     port PATH=~/app/vasp.5.4.4.fix/bin/:$PATH;ulimit -s 512000
                     0;source /beegfs/jhinno/unischeduler/conf/unisched;#######
                     #################################################;#   $JH_
                     NCPU:         Number of CPU cores              #;#   $JH_H
                     OSTFILE:     List of computer hostfiles       #;##########
                     ##############################################;mpirun -np
                     $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log>
Sat Dec 03 19:26:52: Submitted from host <mu01>, CWD </beegfs/home/wangchangxin
                     /data/paper2_add/Ti/H_add_relax/S0/00>, Output File <out.1
                     06971>, Error File <err.106971>, 64 Processors Requested,
                     Requested Resource <span[ptile=64]>;
Sat Dec 03 20:03:52: Exited. Error code 10014. The CPU time used is 0 seconds.

 SCHEDULING PARAMETERS:
              r15s     r1m     r5m    r15m      ut      pg      io      ls
 LoadSched       -       -       -       -       -       -       -       -
 LoadStop        -       -       -       -       -       -       -       -

                it     tmp    swap     mem
 LoadSched       -       -       -       -
 LoadStop        -       -       -       -

-------------------------------------------------------------------------------

Job <106949>, Job Name <skk>, User <wangchangxin>, Project <default>, Status <DO
                     NE>, Queue <cpu>, Application <default>, APS Priority <50
                     00>, Command <#!/bin/sh;#JSUB -J skk;#JSUB -n 64;#JSUB -R
                     span[ptile=64];#JSUB -q cpu;#JSUB -o out.%J;#JSUB -e err.%
                     J;source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64;s
                     ource ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
                     ;source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64;ex
                     port PATH=~/app/vasp.5.4.4.fix/bin/:$PATH;ulimit -s 512000
                     0;source /beegfs/jhinno/unischeduler/conf/unisched;#######
                     #################################################;#   $JH_
                     NCPU:         Number of CPU cores              #;#   $JH_H
                     OSTFILE:     List of computer hostfiles       #;##########
                     ##############################################;mpirun -np
                     $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log>
Sat Dec 03 19:26:52: Submitted from host <mu01>, CWD </beegfs/home/wangchangxin
                     /data/paper2_add/Ti/H_add_relax/S0/00>, Output File <out.1
                     06971>, Error File <err.106971>, 64 Processors Requested,
                     Requested Resource <span[ptile=64]>;
Sat Dec 03 20:03:52: Done successfully. The CPU time used is 2.4433e+06 seconds.
Sat Dec 03 20:02:52: Started on 64. The CPU time used is 2.4433e+06 seconds.

 SCHEDULING PARAMETERS:
              r15s     r1m     r5m    r15m      ut      pg      io      ls
 LoadSched       -       -       -       -       -       -       -       -
 LoadStop        -       -       -       -       -       -       -       -

                it     tmp    swap     mem
 LoadSched       -       -       -       -
 LoadStop        -       -       -       -

-------------------------------------------------------------------------------

Job <106559>, Job Name <skk>, User <wangchangxin>, Project <default>, Status <RUN
                     >, Queue <cpu>, Application <default>, APS Priority <50
                     00>, Command <#!/bin/sh;#JSUB -J skk;#JSUB -n 64;#JSUB -R
                     span[ptile=64];#JSUB -q cpu;#JSUB -o out.%J;#JSUB -e err.%
                     J;source ~/intel/oneapi/mkl/2022.0.2/env/vars.sh intel64;s
                     ource ~/intel/oneapi/compiler/2022.0.2/env/vars.sh intel64
                     ;source ~/intel/oneapi/mpi/2021.5.1/env/vars.sh intel64;ex
                     port PATH=~/app/vasp.5.4.4.fix/bin/:$PATH;ulimit -s 512000
                     0;source /beegfs/jhinno/unischeduler/conf/unisched;#######
                     #################################################;#   $JH_
                     NCPU:         Number of CPU cores              #;#   $JH_H
                     OSTFILE:     List of computer hostfiles       #;##########
                     ##############################################;mpirun -np
                     $JH_NCPU -machinefile $JH_HOSTFILE vasp_std  > vasp.log>
Sat Dec 03 19:26:52: Submitted from host <mu01>, CWD </beegfs/home/wangchangxin
                     /data/paper2_add/Ti/H_add_relax/S0/00>, Output File <out.1
                     06971>, Error File <err.106971>, 64 Processors Requested,
                     Requested Resource <span[ptile=64]>;
Sat Dec 03 20:02:52: Started on 64. The CPU time used is 2.4433e+06 seconds.

 SCHEDULING PARAMETERS:
              r15s     r1m     r5m    r15m      ut      pg      io      ls
 LoadSched       -       -       -       -       -       -       -       -
 LoadStop        -       -       -       -       -       -       -       -

                it     tmp    swap     mem
 LoadSched       -       -       -       -
 LoadStop        -       -       -       -
"""


    sout = stdout.split("-------------------------------------------------------------------------------")

    for line in sout:

        m = re.search(r"Job <([0-9]*)>", line)      #pylint: disable=invalid-name
        if m:
            jobstatus = dict()
            jobstatus["jobid"] = m.group(1).split(".")[0]
            jobstatus["qstatstr"] = line

            m2 = re.search(r"\s*Job Name <(\S*)>,", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            m3 = re.search(r"([0-9]*) Processors Requested", line)  # pylint: disable=invalid-name
            if m3:
                jobstatus["nodes"] = None
                jobstatus["procs"] = m3.group(1)

            jobstatus["walltime"] = None

            m5 = re.search(r"(.*): Started on", line)  # pylint: disable=invalid-name
            if m5:
                ti_str = m5.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["starttime"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))
            else:
                jobstatus["starttime"] = None

            m6 = re.search(r"Status <(\D*\n?\D*)>, Queue", line)  # pylint: disable=invalid-name
            if m6:
                my_status = m6.group(1)
                my_status = my_status.replace("\n", "")
                my_status = my_status.replace(" ", "")

                if my_status == "RUN":
                    jobstatus["jobstatus"] = "R"
                elif my_status == "DONE":
                    jobstatus["jobstatus"] = "C"
                elif my_status == "EXIT":
                    jobstatus["jobstatus"] = "E"
                elif my_status == "PEND":
                    jobstatus["jobstatus"] = "Q"
                else:
                    jobstatus["jobstatus"] = "?"
            else:
                jobstatus["jobstatus"] = None

            if jobstatus["jobstatus"] == "R" and jobstatus["starttime"] != None:
                jobstatus["elapsedtime"] = int(time.time()) - jobstatus["starttime"]
            else:
                jobstatus["elapsedtime"] = None

            m7 = re.search(r"(.*): Done successfully", line)  # pylint: disable=invalid-name
            if m7:
                ti_str = m7.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["completiontime"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            m7 = re.search(r"(.*): Exited", line)  # pylint: disable=invalid-name
            if m7:
                ti_str = m7.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["completiontime"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            m8 = re.search("CWD <(.*)>, Out", line)
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            m9 = re.search(r"(.*): Submitted from host", line)  # pylint: disable=invalid-name
            if m9:
                ti_str = m9.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["subtime"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            status[jobstatus["jobid"]] = jobstatus

    return status

def submit(substr):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """
    m = re.search(r"-J\s+(.*)\s", substr)       #pylint: disable=invalid-name
    if m:
        jobname = m.group(1)        #pylint: disable=unused-variable
    else:
        raise PBSError(
            None,
            r"Error in pbs.misc.submit(). Jobname (\"-N\s+(.*)\s\") not found in submit string.")

    p = subprocess.Popen(   #pylint: disable=invalid-name
        "jsub <", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate(input=substr)       #pylint: disable=unused-variable
    stdout = stdout.decode()
    stderr = stderr.decode()
    print(stdout[:-1])

    if re.search("error", stdout):
        raise PBSError(0, "PBS Submission error.\n" + stdout + "\n" + stderr)
    else:
        jobid = stdout.split(".")[0]
        return jobid

def delete(jobid):
    """qdel a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["jctrl", "kill", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()        #pylint: disable=unused-variable
    return p.returncode

def hold(jobid):
    """qhold a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["jctrl", "stop", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode

def release(jobid):
    """qrls a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["jctrl", "resume", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode


if __name__=="__main__":
    # res21 = find_executable("qsub")
    res = job_status(jobid=None)
    print(res)
""" Misc functions for interfacing between torque and the pbs module """

#pylint: disable=line-too-long, too-many-locals, too-many-branches

### External ###
import subprocess
import os

import re
import datetime
import time
# import sys

### Internal ###
from misc import getversion, getlogin, seconds, PBSError, run_cmd


def _squeue(jobid=None, username=getlogin(), full=False, version=getversion(), sformat=None):    #pylint: disable=unused-argument
    """Return the stdout of squeue minus the header lines.

       By default, 'username' is set to the current user.
       'full' is the '-f' option
       'jobid' is a string or list of strings of job ids
       'version' is a software version number, used to
            determine compatible ops
       'sformat' is a squeue format string (e.g., "%A %i %j %c")

       Returns the text of squeue, minus the header lines
    """

    # If Full is true, we need to use scontrol:
    if full is True:
        if jobid is None:
            if username is None:
                # Clearly we want ALL THE JOBS
                sopt = ["scontrol", "show", "job"]

                sout = run_cmd(sopt)

                # Nothing to strip, as scontrol provides no headers
                return sout

            else:
                # First, get jobids that belong to that username using
                # squeue (-h strips the header)
                sopt = ["scontrol", "show", "job", "-u", username]

                sout = run_cmd(sopt)

                # Nothing to strip, as scontrol provides no headers
                return sout

        # Ensure the jobids are a list, even if they're a list of 1...
        if not isinstance(jobid, list) and jobid is not None:
            jobid = [jobid]
        if isinstance(jobid, list):
            opt = ["scontrol", "show", "job"]
            sreturn = ""
            for my_id in jobid:
                sopt = opt + [str(my_id)]

                sout = run_cmd(sopt)

                sreturn = sreturn + sout + "\n\n"

            return sreturn

    else:
        sopt = ["squeue", "-h"]
        if username is not None:
            sopt += ["-u", username]
        if jobid is not None:
            sopt += ["--job="]
            if isinstance(jobid, list):
                sopt += ["'"+",".join([str(i) for i in jobid])+"'"]
            else:
                sopt += [str(jobid)]
        if sformat is not None:
            sopt += ["-o", "'" + sformat + "'"]
        else:
            if jobid is None and username is None:
                sopt += ["-o", "'%i %u %P %j %U %D %C %m %l %t %M'"]
            else:
                sopt += ["-o", "'%i %j %u %M %t %P'"]

        sout = run_cmd(sopt)

        # return the remaining text
        return sout

def job_id(all=False, name=None):       #pylint: disable=redefined-builtin
    """If 'name' given, returns a list of all jobs with a particular name using squeue.
       Else, if all=True, returns a list of all job ids by current user.
       Else, returns this job id from environment variable PBS_JOBID (split to get just the number).

       Else, returns None

    """
    if all or name is not None:
        jobid = []
        stdout = _squeue()
        sout = stdout
        for line in sout:
            if name is not None:
                if line.split()[3] == name:
                    jobid.append((line.split()[0]).split(".")[0])
            else:
                jobid.append((line.split()[0]).split(".")[0])
        return jobid
    else:
        if 'SLURM_JOBID' in os.environ:
            return os.environ['SLURM_JOBID'].split(".")[0]
        else:
            return None
            #raise PBSError(
            #    "?",
            #    "Could not determine jobid. 'PBS_JOBID' environment variable not found.\n"
            #    + str(os.environ))

def job_rundir(jobid):
    """Return the directory job "id" was run in using squeue.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if isinstance(jobid, (list,tuple)):
        for i in jobid:
            stdout = _squeue(jobid=i, full=True)
            match = re.search("WorkDir=(.*)\n", stdout)
            rundir[i] = match.group(1)
    else:
        stdout = _squeue(jobid=jobid, full=True)
        match = re.search("WorkDir=(.*)\n", stdout)
        rundir[jobid] = match.group(1)
    return rundir

def job_status(jobid=None):
    """Return job status using squeue

       Returns a dict of dict, with jobid as key in outer dict.
       Inner dict contains:
       "name", "nodes", "procs", "walltime",
       "jobstatus": status ("Q","C","R", etc.)
       "qstatstr": result of squeue -f jobid, None if not found
       "elapsedtime": None if not started, else seconds as int
       "starttime": None if not started, else seconds since epoch as int
       "completiontime": None if not completed, else seconds since epoch as int

       *This should be edited to return job_status_dict()'s*
    """
    status = dict()

    stdout = _squeue(jobid=jobid, full=True)
    sout = stdout.split("\n\n")

    # jobstatus = None

    for line in sout:

        m = re.search(r"Job\s?Id\s?=\s?(.*)\sJ", line)      #pylint: disable=invalid-name
        if m:
            jobstatus = {"jobid" : None, "nodes" : None, "procs" : None,
                         "walltime" : None, "qstatstr" : None, "elapsedtime" : None,
                         "starttime" : None, "completiontime" : None, "jobstatus" : None, "cluster": None}

            jobstatus["jobid"] = m.group(1)
            jobstatus["qstatstr"] = line

            m2 = re.search(r"JobName=\s?(.*)\n", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            # Look for the Nodes/PPN Info
            m3 = re.search(r"NumNodes=\s?([0-9]*-[0-9]*)\s", line)  # pylint: disable=invalid-name
            if m3:
                jobstatus["nodes"] = m3.group(1)
                m4 = re.search(r"NumCPUs=\s?([0-9]*)\s", line)  # pylint: disable=invalid-name
                if m4:
                    jobstatus["procs"] = int(m4.group(1))

            # Grab the job start time
            m4 = re.search(r"StartTime=\s?([0-9]*\-[0-9]*\-[0-9]*T[0-9]*:[0-9]*:[0-9]*)\s",
                          line)  # pylint: disable=invalid-name
            if m4:
                if m4.group(1) != "Unknown":
                    year, month, day = m4.group(1).split("T")[0].split("-")
                    hrs, mns, scs = m4.group(1).split("T")[1].split(":")
                    starttime = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hrs),
                                                  minute=int(mns), second=int(scs))
                    jobstatus["starttime"] = time.mktime(starttime.timetuple())

            m8 = re.search("WorkDir=\s?(.*)", line)
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            # Look for timing info
            m9 = re.search(r"RunTime=\s?([0-9]*:[0-9]*:[0-9]*)\s", line) #pylint: disable=invalid-name
            if m9:
                if m9.group(1) != "Unknown":
                    hrs, mns, scs = m9.group(1).split(":")
                    runtime = datetime.timedelta(hours=int(hrs), minutes=int(mns), seconds=int(scs))
                    jobstatus["elapsedtime"] = runtime.seconds

                    m10 = re.search(r"TimeLimit=\s?([0-9]*:[0-9]*:[0-9]*)\s", line) #pylint: disable=invalid-name
                    if m10:
                        walltime = datetime.timedelta(hours=int(hrs), minutes=int(mns), seconds=int(scs))
                        jobstatus["walltime"] = walltime.seconds

            # Grab the job status
            m11 = re.search(r"JobState=\s?([a-zA-Z]*)\s", line) #pylint: disable=invalid-name
            if m11:
                my_status = m11.group(1)
                if my_status == "RUNNING" or my_status == "CONFIGURING":
                    jobstatus["jobstatus"] = "R"
                elif my_status == "BOOT_FAIL" or my_status == "FAILED" or my_status == "NODE_FAIL" or my_status == "CANCELLED" or my_status == "COMPLETED" or my_status == "PREEMPTED" or my_status == "TIMEOUT":
                    jobstatus["jobstatus"] = "C"
                elif my_status == "COMPLETING" or my_status == "STOPPED":
                    jobstatus["jobstatus"] = "E"
                elif my_status == "PENDING" or my_status == "SPECIAL_EXIT":
                    jobstatus["jobstatus"] = "Q"
                elif my_status == "SUSPENDED":
                    jobstatus["jobstatus"] = "S"
                else:
                    jobstatus["jobstatus"] = "?"

            if jobstatus["jobstatus"]=="C":
                m13 = re.search(r"EndTime=\s?([0-9]*\-[0-9]*\-[0-9]*T[0-9]*:[0-9]*:[0-9]*)\s",
                               line)  # pylint: disable=invalid-name
                if m13:
                    year, month, day = m13.group(1).split("T")[0].split("-")
                    hrs, mns, scs = m13.group(1).split("T")[1].split(":")
                    starttime = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hrs),
                                                  minute=int(mns), second=int(scs))
                    jobstatus["completiontime"] = time.mktime(starttime.timetuple())

            # Grab the cluster/allocating node:
            m12 = re.search(r"AllocNode:.*=\s?(.*):.*\s", line) #pylint: disable=invalid-name
            if m12:
                raw_str = m12.group(1)
                m12 = re.search(r"(.*?)(?=[^a-zA-Z0-9]*login.*)\s", raw_str)    #pylint: disable=invalid-name
                if m12:
                    jobstatus["cluster"] = m12.group(1)
                else:
                    jobstatus["cluster"] = raw_str

            m9 = re.search(r"SubmitTime=\s?([0-9]*:[0-9]*:[0-9]*)\s", line)  # pylint: disable=invalid-name
            if m9:
                jobstatus["subtime"] = int(time.mktime(datetime.datetime.strptime(
                    m9.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            status[jobstatus["jobid"]] = jobstatus

    return status

def submit(substr):
    """Submit a PBS job using sbatch.

       substr: The submit script string
    """
    m = re.search(r"-J\s+(.*)\s", substr)       #pylint: disable=invalid-name
    if m:
        jobname = m.group(1)        #pylint: disable=unused-variable
    else:
        raise PBSError(
            0,
            r"Error in pbs.misc.submit(). Jobname (\"-N\s+(.*)\s\") not found in submit string.")
    
    p = subprocess.Popen(   #pylint: disable=invalid-name
        "sbatch", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate(input=substr) #pylint: disable=unused-variable
    stdout = stdout.decode()
    stderr = stderr.decode()
    print(stdout[:-1])
    if re.search("error", stdout):
        raise PBSError(0, "PBS Submission error.\n" + stdout + "\n" + stderr)
    else:
        jobid = stdout.rstrip().split()[-1]
        return jobid


def submit_file(file):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """

    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["sbatch <", file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()       #pylint: disable=unused-variable
    stdout = stdout.decode()
    stderr = stderr.decode()
    print(stdout[:-1])

    if re.search("error", stdout):
        raise PBSError(0, "PBS Submission error.\n" + stdout + "\n" + stderr)
    else:
        jobid = stdout.split(".")[0]
        return jobid
def delete(jobid):
    """scancel a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["scancel", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()        #pylint: disable=unused-variable
    return p.returncode

def hold(jobid):
    """scontrol delay a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["scontrol", "update", "JobId=", jobid, "StartTime=", "now+30days"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode

def release(jobid):
    """scontrol un-delay a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["scontrol", "update", "JobId=", jobid, "StartTime=", "now"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode

def alter(jobid, arg):
    """scontrol update PBS job.

        'arg' is a pbs command option string. For instance, "-a 201403152300.19"
    """
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["scontrol", "update", "JobId=", jobid] + arg.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode


if __name__=="__main__":
    # res21 = find_executable("qsub")
    res = job_status()
    print(res[list(res.keys())[-1]])
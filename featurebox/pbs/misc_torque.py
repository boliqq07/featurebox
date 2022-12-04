""" Misc functions for interfacing between torque and the pbs module """

import subprocess
import os
import re
import datetime
import time
import sys
from misc import getversion, getlogin, seconds, PBSError, run_cmd


def _qstat(jobid=None, username=getlogin(), full=False, version=getversion()):
    """Return the stdout of qstat minus the header lines.

       By default, 'username' is set to the current user.
       'full' is the '-f' option
       'id' is a string or list of strings of job ids

       Returns the text of qstat, minus the header lines
    """

    # -u and -f contradict in earlier versions of Torque
    if full and username is not None and int(version.split('.')[0]) < 5 and jobid is None:
        # First get all jobs by the user
        qopt = ["qselect"]
        qopt += ["-u", username]

        # Call 'qselect' using subprocess

        qsout = run_cmd(qopt)

        # Get the jobids
        jobid = []
        for line in qsout:
            jobid += [line.rstrip("\n")]

    opt = ["qstat"]
    # If there are jobid(s), you don't need a username
    if username is not None and jobid is None:
        opt += ["-u", username]
    # But if there are jobid(s) and a username, you need -a to get full output
    elif username is not None and jobid is not None and not full:
        opt += ["-a"]
    # By this point we're guaranteed torque ver >= 5.0, so -u and -f are safe together
    if full:
        opt += ["-f"]
    if jobid is not None:
        if isinstance(jobid, str):
            jobid = [jobid]
        elif isinstance(jobid, list):
            pass
        else:
            print("Error in pbs.misc.qstat(). type(jobid):", type(jobid))
            sys.exit()
        opt += jobid

    # call 'qstat' using subprocess
    # print opt
    sout = run_cmd(opt)

    # strip the header lines
    if full is False:
        for line in sout:
            if line[0] == "-":
                break

    # return the remaining text
    return sout

def job_id(all=False, name=None):       #pylint: disable=redefined-builtin
    """If 'name' given, returns a list of all jobs with a particular name using qstat.
       Else, if all=True, returns a list of all job ids by current user.
       Else, returns this job id from environment variable PBS_JOBID (split to get just the number).

       Else, returns None

    """
    if all or name is not None:
        jobid = []
        stdout = _qstat()
        sout = stdout.splitlines()
        for line in sout:
            if name is not None:
                if line.split()[3] == name:
                    jobid.append((line.split()[0]).split(".")[0])
            else:
                jobid.append((line.split()[0]).split(".")[0])
        return jobid
    else:
        if 'PBS_JOBID' in os.environ:
            return os.environ['PBS_JOBID'].split(".")[0]
        else:
            return None
            #raise PBSError(
            #    "?",
            #    "Could not determine jobid. 'PBS_JOBID' environment variable not found.\n"
            #    + str(os.environ))

def job_rundir(jobid):
    """Return the directory job "id" was run in using qstat.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if isinstance(jobid, (list,tuple)):
        for i in jobid:
            stdout = _qstat(jobid=i, full=True)
            match = re.search("init_work_dir = (.*),", stdout)
            rundir[i] = match.group(1)
    else:
        stdout = _qstat(jobid=jobid, full=True)
        match = re.search("init_work_dir = (.*),", stdout)
        rundir[jobid] = match.group(1)
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

    # sout = stdout.splitlines()
    sout = stdout.split("\n\n")

    for line in sout:

        m = re.search(r"Job Id:\s*(.*)", line)      #pylint: disable=invalid-name
        if m:
            jobstatus = dict()
            jobstatus["jobid"] = m.group(1).split(".")[0]
            jobstatus["qstatstr"] = line

            m2 = re.search(r"\s*Job_Name\s*=\s*(.*)\s", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            m3 = re.search(r"\s*Resource_List\.nodes\s*=\s*(.*):ppn=(.*)\s", line)  # pylint: disable=invalid-name
            if m3:
                jobstatus["nodes"] = m3.group(1)
                jobstatus["procs"] = int(m3.group(1)) * int(m3.group(2))

            m4 = re.search(r"\s*Resource_List\.walltime\s*=\s*(.*)\s", line)  # pylint: disable=invalid-name
            if m4:
                jobstatus["walltime"] = int(seconds(m4.group(1)))

            m5 = re.search(r"\s*start_time\s*=\s*(.*)\s", line)  # pylint: disable=invalid-name
            if m5:
                jobstatus["starttime"] = int(time.mktime(datetime.datetime.strptime(
                    m5.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))
            else:
                jobstatus["starttime"] = None

            m6 = re.search(r"\s*job_state\s*=\s*(.*)\s", line)  # pylint: disable=invalid-name
            if m6:
                jobstatus["jobstatus"] = m6.group(1)
            else:
                jobstatus["jobstatus"] = None

            if jobstatus["jobstatus"] == "R" and jobstatus["starttime"] != None:  # pylint: disable=unsubscriptable-object
                jobstatus["elapsedtime"] = int(time.time()) - jobstatus[
                    "starttime"]  # pylint: disable=unsubscriptable-object
            else:
                jobstatus["elapsedtime"] = None


            m7 = re.search(r"\s*comp_time\s*=\s*(.*)\s", line)  # pylint: disable=invalid-name
            if m7:
                jobstatus["completiontime"] = int(time.mktime(datetime.datetime.strptime(
                    m7.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            m8 = re.search("init_work_dir = (.*)", line)
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            m9 = re.search(r"\s*ctime\s*=\s*(.*)\s", line)  # pylint: disable=invalid-name
            if m9:
                jobstatus["subtime"] = int(time.mktime(datetime.datetime.strptime(
                    m9.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            status[jobstatus["jobid"]] = jobstatus

    return status

def submit(substr):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """
    m = re.search(r"-N\s+(.*)\s", substr)       #pylint: disable=invalid-name
    if m:
        jobname = m.group(1)        #pylint: disable=unused-variable
    else:
        raise PBSError(
            None,
            r"Error in pbs.misc.submit(). Jobname (\"-N\s+(.*)\s\") not found in submit string.")

    p = subprocess.Popen(   #pylint: disable=invalid-name
        "qsub", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate(input=substr)       #pylint: disable=unused-variable
    stdout = stdout.decode()
    stderr = stderr.decode()
    print(stdout[:-1])

    if re.search("error", stdout):
        raise PBSError(0, "PBS Submission error.\n" + stdout + "\n" + stderr)
    else:
        jobid = stdout.split(".")[0]
        return jobid

def submit_file(file):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """

    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["qsub",  file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    """qdel a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["qdel", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()        #pylint: disable=unused-variable
    return p.returncode

def hold(jobid):
    """qhold a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["qhold", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode

def release(jobid):
    """qrls a PBS job."""
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["qrls", jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode

def alter(jobid, arg):
    """qalter a PBS job.

        'arg' is a pbs command option string. For instance, "-a 201403152300.19"
    """
    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["qalter"] + arg.split() + [jobid], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()    #pylint: disable=unused-variable
    return p.returncode


if __name__=="__main__":
    # res21 = find_executable("qsub")
    res = job_status()
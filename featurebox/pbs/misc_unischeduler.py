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

    stdout = _jjobs(jobid=jobid, full=True)


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

            m10 = re.search(r"ORDER: (.*)", line)  # pylint: disable=invalid-name
            if m10:
                jobstatus["order"] = m10.group(1)

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

def submit_file(file):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """

    p = subprocess.Popen(   #pylint: disable=invalid-name
        ["jsub <",  file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
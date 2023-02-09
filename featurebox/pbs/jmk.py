# -*- coding: utf-8 -*-

# @Time  : 2022/12/7 8:48
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
# -*- coding: utf-8 -*-

from featurebox.pbs.job_manager import JobManager

_dos_help = """
-------------------------------
Job manager command line model
-------------------------------

Example:

1. get the work path.

    jmk 734
    
2. cd the work path.
    
    cd $(jmk 734)

"""


def run(args, parser):
    jm = JobManager(manager=None)
    jm.get_job_msg()
    idd = str(args.idd)

    if idd in jm.msg:
        try:
            pt = str(jm.msg[idd]["work_dir"])
        except:
            print("Please offer one exist id of job.")
            pt = "."
    else:
        print("Please offer one exist id of job.")
        pt = "."
    print(pt)
    return pt


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='idd', help='id of job', type=str, default=None)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        return run(args, parser)


def main():
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """

    from mgetool.cli._formatter import Formatter
    import argparse

    parser = argparse.ArgumentParser(description=_dos_help, formatter_class=Formatter)
    CLICommand.add_arguments(parser=parser)
    args = CLICommand.parse_args(parser=parser)
    return CLICommand.run(args=args, parser=parser)



if __name__ == '__main__':
    main()

import os
import re

def _set_bachrc():

    res1 = os.popen("whereis sbatch").readlines()[0]
    res2 = os.popen("whereis jsub").readlines()[0]
    res3 = os.popen("whereis qsub").readlines()[0]

    res11 = re.search("(/\S+)+",res1)
    if res11:
        return "sbatch",res11.group()

    res22 = re.search("(/\S+)+",res2)
    if res22:
        return "jsub",res22.group()

    res33 = re.search("(/\S+)+",res3)
    if res33:
        return "qsub",res33.group()


def set_bachrc(path = "~/history_jobs"):
    a, b = _set_bachrc()

    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(f"{path}/paths.temp"):
        with open(f"{path}/paths.temp", "w") as f:
            f.write("")

    text = f'alias {a}="pwd >> {path}/paths.temp|{b}"/n'

    with open("~/.bashrc", "w+") as f:

        wods = f.readlines()
        if text in wods:
            pass
        else:
            wods.append(text)
            f.writelines(wods)


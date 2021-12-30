from mgetool.imports import BatchFile

bf = BatchFile("/home/wcx/data/NM-scf-dos/")
bf.filter_dir_name(include="base", layer=-1)
bf.merge()
fdir = bf.file_dir

from featurebox.cli.bgefvk import cal_all
result_all = cal_all(fdir, store=True, store_name="bandgap_Ef.csv", run_cmd=False)

from featurebox.cli.dbcvk import cal_all
result_all = cal_all(fdir, store=True, store_name="dbc.csv", run_cmd=False)


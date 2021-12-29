#!/bin/sh
#SBATCH --partition=normal
#SBATCH --job-name=smh
#SBATCH --nodes=2
#SBATCH --ntasks=28

source /data/home/suyj/intel/bin/compilervars.sh intel64
export PATH=/data/home/suyj/app/vasp.5.4.4-2018/bin:$PATH
#mpirun -np $SLURM_NPROCS vasp_std | tee output
mpirun -np 56 vasp_std | tee output





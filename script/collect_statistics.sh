#! /bin/bash
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH --ntasks 1
# SBATCH --cpus-per-task 1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 01:00:00
#SBATCH -p hpc4-3d

$MPIRUN python collect_statistics.py

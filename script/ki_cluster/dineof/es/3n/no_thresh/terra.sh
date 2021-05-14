#! /bin/bash
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 02-23:59:59
#SBATCH -p hpc4-3d

module load openmpi intel-compilers
export OPENBLAS_NUM_THREADS=1


### es - 3n - no thresh
$MPIRUN python main3_mp.py -c config/main3_default_ki_cluster.yml \
    -S terra \
    --logs ../test/reconstruction_logs/dineof_es_3neighbours_terra \
    --interpolated-stem interpolated_3neighbours \
    --output-stem Output_3neighbours \
    --decomposition-method DINEOF \
    --early-stopping 1

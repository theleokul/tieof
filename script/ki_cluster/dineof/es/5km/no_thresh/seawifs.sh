#! /bin/bash
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH -n 16
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 02-23:59:59
#SBATCH -p hpc4-3d

module load openmpi intel-compilers
export OPENBLAS_NUM_THREADS=8


### es - 5km - no thresh
$MPIRUN python main3_mp.py -c config/main3_default_ki_cluster.yml \
    -S seawifs \
    --logs ../test/reconstruction_logs/dineof_es_5kmradius_seawifs \
    --interpolated-stem interpolated_5kmradius \
    --output-stem Output_5kmradius \
    --decomposition-method DINEOF \
    --early-stopping 1

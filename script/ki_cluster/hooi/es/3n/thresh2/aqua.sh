#! /bin/bash
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 02-23:59:59
#SBATCH -p hpc4-3d

module load openmpi intel-compilers
export OPENBLAS_NUM_THREADS=2


### es - 3n - no thresh
$MPIRUN python main3_mp.py -c config/main3_default_ki_cluster.yml \
    --satellite-descriptor '../test/satellite_descriptor_ki_cluster_w3nt2.csv' \
    -S aqua \
    --logs ../test/reconstruction_logs/hooi_es_3neighbours_thresh2_aqua \
    --interpolated-stem interpolated_3neighbours_thresh2 \
    --output-stem Output_3neighbours_thresh2 \
    --decomposition-method hooi \
    --early-stopping 1

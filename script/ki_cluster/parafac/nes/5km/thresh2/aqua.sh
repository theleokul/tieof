#! /bin/sh
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 02-23:59:59
#SBATCH -p hpc4-3d

module load openmpi intel-compilers
export OPENBLAS_NUM_THREADS=2

### es - 5km - thresh2
$MPIRUN python main3_mp.py -c config/main3_default_ki_cluster.yml \
    -S aqua \
    --logs ../test/reconstruction_logs/parafac_nes_5kmradius_thresh2_aqua \
    --interpolated-stem interpolated_5kmradius_thresh2 \
    --output-stem Output_5kmradius_thresh2 \
    --decomposition-method PARAFAC \
    --early-stopping 0

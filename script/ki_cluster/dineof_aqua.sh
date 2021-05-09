#! /bin/bash
#SBATCH -D /s/ls4/users/leokul01/dineof3/script
#SBATCH -n 1
#SBATCH --cpus-per-task 19
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 02-23:59:59
#SBATCH -p hpc4-3d

module load openmpi intel-compilers

### es - 5km - thresh2
$MPIRUN python main3_mp.py -c config/main3_default.yml \
    -S aqua \
    --logs ../test/reconstruction_logs/dineof_es_5kmradius_thresh2_aqua \
    --interpolated-stem interpolated_5kmradius_thresh2 \
    --output-stem Output_5kmradius_thresh2 \
    --decomposition-method DINEOF \
    --early-stopping 1

### es - 5km - no thresh
$MPIRUN python main3_mp.py -c config/main3_default.yml \
    -S aqua \
    --logs ../test/reconstruction_logs/dineof_es_5kmradius_aqua \
    --interpolated-stem interpolated_5kmradius \
    --output-stem Output_5kmradius \
    --decomposition-method DINEOF \
    --early-stopping 1

### es - 3n - no thresh
$MPIRUN python main3_mp.py -c config/main3_default.yml \
    -S aqua \
    --logs ../test/reconstruction_logs/dineof_es_3neighbours_aqua \
    --interpolated-stem interpolated_3neighbours \
    --output-stem Output_3neighbours \
    --decomposition-method DINEOF \
    --early-stopping 1

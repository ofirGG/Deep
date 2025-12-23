#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="test_job"
MAIL_USER="kerenmizrahi@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
    
#<<EOF

#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
#python $@

# Our code:
#==========================================================
# -----------------Experiment 1.1-------------------------- 

# K=32 fixed, with L=2,4,8,16 varying per run
K=32
for L in 2 4 8 16; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_1 -K $K -L $L -P 6 -H 256 64 16 -d cuda --reg 0.0001 --lr 0.0003
done

# K=64 fixed, with L=2,4,8,16 varying per run
K=64
for L in 2 4 8 16; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_1 -K $K -L $L -P 6 -H 256 64 16 -d cuda --reg 0.0001 --lr 0.0003
done

#-----------------Experiment 1.2------------------------- 

#L=2 fixed, with K=[32],[64],[128] varying per run.
L=2
for K in 32 64 128; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_2 -K $K -L $L -P 6 -H 256 64 16 -d cuda --reg 0.001 --lr 0.0003
done

#L=4 fixed, with K=[32],[64],[128] varying per run.
L=4
for K in 32 64 128; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_2 -K $K -L $L -P 6 -H 256 64 16 -d cuda --reg 0.001 --lr 0.0003
done

#L=8 fixed, with K=[32],[64],[128] varying per run.
L=8
for K in 32 64 128; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_2 -K $K -L $L -P 6 -H 256 64 16 -d cuda --reg 0.001 --lr 0.0001
done

# srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_2 -K 32 -L 8 -P 6 -H 256 64 16 -d cuda --reg 0.001 --lr 0.0003 

#-----------------Experiment 1.3-----------------------

# K=[64, 128] fixed with L=2,3,4 varying per run
for L in 2 3 4; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L $L -P 3 -H 256 64 16 -d cuda --reg 0.001 --lr 0.0002
done

#-----------------Experiment 1.4-----------------------

#K=[32] fixed with L=8,16,32 varying per run
for L in 8 16 32; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_4 -K 32 -L $L -P 6 -H 256 64 16 -d cuda -M resnet --lr 0.0005 --reg 0.001
done

#K=[64, 128, 256] fixed with L=2,4,8 varying per run
for L in 2 4 8; do
    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L $L -P 6 -H 256 64 16 -d cuda -M resnet --reg 0.001 --lr 0.0005
done

#=========================================================

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
#EOF


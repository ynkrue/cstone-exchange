#!/bin/bash
#SBATCH --job-name=relay-gpu
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.9:3103280092
#SBATCH --view=default
#SBATCH --output=/users/yrfenach/cstone-exchange/logs/%x-%j.out

# GPU test on 4, 8, ... nodes up to the allocation (P 16 up), 4 ranks per node with one GPU each
export MPICH_GPU_IPC_THRESHOLD=1
export MPIR_CVAR_ENABLE_NODE_COMM=1

BIN=${1:-build/test_global_exchange_gpu}

for ((n = 4; n <= SLURM_JOB_NUM_NODES; n *= 2)); do
    srun -N$n --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=72 -t2 $BIN
done

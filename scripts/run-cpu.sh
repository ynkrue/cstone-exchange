#!/bin/bash
#SBATCH --job-name=relay-cpu
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.9:3103280092
#SBATCH --view=default
#SBATCH --output=/users/yrfenach/cstone-exchange/logs/%x-%j.out

# CPU test on P 16, 32, ... up to 64 ranks per allocated node, nodes filled with up to 64 ranks each
BIN=${1:-build/test_global_exchange}

for ((p = 16; p <= 64 * SLURM_JOB_NUM_NODES; p *= 2)); do
    srun -n$p -N$(((p + $SLURM_NTASKS_PER_NODE-1) / $SLURM_NTASKS_PER_NODE)) --cpus-per-task=4 -t3 $BIN
done

#!/bin/bash
#SBATCH --job-name=cstone-exchange
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --output=/users/yrfenach/cstone-exchange/logs/%x-%j.out

export MPICH_GPU_IPC_THRESHOLD=1

srun -N4 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 0
srun -N4 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 1
srun -N4 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 2

srun -N8 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 0
srun -N8 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 1
srun -N8 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 2

srun -N16 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 0
srun -N16 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 1
srun -N16 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 2

srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 0
srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 1
srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -c1 -t1 build/test_global_exchange_gpu 2
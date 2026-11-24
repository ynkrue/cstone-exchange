#!/bin/bash
#SBATCH --job-name=cstone-exchange-cpu
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --output=/users/yrfenach/cstone-exchange/logs/%x-%j.out

srun -N1 --ntasks-per-node=16 build/test_global_exchange
srun -N2 --ntasks-per-node=16 build/test_global_exchange
srun -N4 --ntasks-per-node=16 build/test_global_exchange
srun -N8 --ntasks-per-node=16 build/test_global_exchange
srun -N16 --ntasks-per-node=16 build/test_global_exchange
srun -N32 --ntasks-per-node=16 build/test_global_exchange
#srun -N64 --ntasks-per-node=16 build/test_global_exchange

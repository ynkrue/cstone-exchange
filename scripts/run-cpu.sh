#!/bin/bash
#SBATCH --job-name=cstone-exchange-cpu
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --output=/users/yrfenach/cstone-exchange/logs/%x-%j.out

srun -N1 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 0
srun -N1 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 1
srun -N1 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 2

srun -N2 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 0
srun -N2 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 1
srun -N2 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 2

srun -N4 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 0
srun -N4 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 1
srun -N4 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 2

srun -N8 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 0
srun -N8 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 1
srun -N8 --ntasks-per-node=64 -c4 -t1 build/test_global_exchange 2

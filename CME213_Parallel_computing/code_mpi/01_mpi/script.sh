#!/bin/bash
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

mpirun -n 4 ./mpi_hello
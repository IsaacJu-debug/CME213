#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=CME
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------
echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $SLURM_SUBMIT_DIR`"
echo
echo Output from code
echo ----------------
### end of information preamble

cd $SLURM_SUBMIT_DIR

mpiexec -np 1 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 2 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 4 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 8 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 16 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 32 ./samplesort -n 120000000
echo
echo "------------"
mpiexec -np 64 ./samplesort -n 120000000

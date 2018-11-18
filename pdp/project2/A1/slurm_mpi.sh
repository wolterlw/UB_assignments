#!/bin/bash

####### select partition (check CCR documentation)
#SBATCH --constraint=CPU-E5645

####### set memory that nodes provide (check CCR documentation, e.g. 48GB for CPU-E5645)
#SBATCH --mem=48000

####### make sure no other jobs are assigned to your nodes
#SBATCH --exclusive

####### further customizations
#SBATCH --job-name="vliunda_pdp_a1"
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=12
#SBATCH --time=00:10:00

####### check modules to see which version of MPI is available
####### and use appropriate module if needed
module load intel-mpi/2017.0.1
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

srun ./a1_test 100000000
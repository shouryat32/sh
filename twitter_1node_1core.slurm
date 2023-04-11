#!/bin/bash
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Total number of tasks (cores)
#SBATCH -t 00:20:00                  # Wall time limit (hh:mm:ss)
#SBATCH -J twitter_1_node_core         # Job name
#SBATCH -o twitter_1_node_core.%j.out  # Output file name
#SBATCH -e twitter_1_node_core.err  # Error file name
# Load required modules
module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong
source ~/virtualenv/python3.7.4/bin/activate
cd ~/assignment_1/sh
mpiexec -np 1 python main.py
deactivate
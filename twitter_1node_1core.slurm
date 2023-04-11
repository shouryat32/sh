#!/bin/bash
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Total number of tasks (cores)
#SBATCH -t 1:00:00                  # Wall time limit (hh:mm:ss)
#SBATCH -J twitter_1_node_core         # Job name
#SBATCH -o twitter_1_node_core.%j.out  # Output file name
#SBATCH -e twitter_1_node_core.err  # Error file name
# Load required modules
module load Python            
mkdir ~/virtualenv
virtualenv ~/virtualenv/python3.8.6
pip install ijson
pip install pandas
pip install mpi4py
source ~/virtualenv/python3.8.6/bin/activate
# Activate your virtual environment if necessary
# source activate my_virtual_env
# Change to your working directory
cd /path/to/your/working/directory
# Run your MPI-based Python script
srun mpiexec -np 1 python main.py
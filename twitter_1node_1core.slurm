
#!/bin/bash
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Total number of tasks (cores)
#SBATCH -t 1:00:00                  # Wall time limit (hh:mm:ss)
#SBATCH -J twitter_1_node_core         # Job name
#SBATCH -o twitter_1_node_core.%j.out  # Output file name
#SBATCH -e twitter_1_node_core.err  # Error file name
# Load required modules
virtualenv ~/project1/python/3.8.6
module load mpi4py/3.0.2-timed-pingpong

# Activate your virtual environment if necessary
# source activate my_virtual_env
# Change to your working directory
# Run your MPI-based Python script
srun mpiexec -np 1 python main.py
deactivate


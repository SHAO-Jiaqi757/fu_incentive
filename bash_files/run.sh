#!/bin/bash
#SBATCH --job-name=my_experiment    # Job name
#SBATCH --output=logs/output_%j.txt      # Output log file (%j will be replaced with the job ID)
#SBATCH --error=logs/error_%j.txt        # Error log file (%j will be replaced with the job ID)
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --time=0-12:00:00             # Time limit hrs:min:sec
#SBATCH --gres=gpu:4                
#SBATCH --partition=gpu-share       # Partition (queue) to submit to (check with your cluster admin for available partitions)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fu_incentive
# Run your script
./run_fl.sh

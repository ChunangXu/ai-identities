#!/bin/bash
#SBATCH --nodes=1                  # Use 1 node
#SBATCH --ntasks-per-node=40       # Use all 40 cores
#SBATCH --cpus-per-task=1          # 1 CPU per task
#SBATCH --mem=188G                 # Use ~188 GiB of memory (max per node)
#SBATCH --time=02:00:00            # Time limit (2 hours)
#SBATCH --account=def-engine14     # Replace with your account
#SBATCH --job-name=llama-8bit-cpu  # Job name
#SBATCH --output=%x-%j.out         # Output file (job name and job ID)
#SBATCH --error=%x-%j.err          # Error file (job name and job ID)

# Load required modules
module purge
module load python/3.10 gcc/9.3.0
module load cuda torch

# Set up environment
export PROJECT=/project/def-engine14/goutham  # Replace with your project directory
export VENV=$PROJECT/llama-env                # Path to virtual environment
export DATA=$PROJECT/llama-data               # Path to data and model weights

# Activate virtual environment
source $VENV/bin/activate

# Run the script
python $PROJECT/inference_8bit.py \
    --model_path $DATA/llama-weights/7B \
    --output_dir $PROJECT/results
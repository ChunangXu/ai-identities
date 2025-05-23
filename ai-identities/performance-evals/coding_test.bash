#!/bin/bash
#SBATCH --ntasks=17
#SBATCH --nodes=17
#SBATCH --cpus-per-task=80          # 1 CPU per task
#SBATCH --time=02:00:00            # Time limit (2 hours)
#SBATCH --account=def-engine14     # Replace with your account
#SBATCH --job-name=llama3.2-mmlu-pro-coding  # Job name
#SBATCH --output=%x-%j.out         # Output file (job name and job ID)
#SBATCH --error=%x-%j.err          # Error file (job name and job ID)
module purge
module load python/3.11.5 gcc/9.3.0
module load cuda torch
export VENV=$SCRATCH/ai-identities/performance-evals/venv                # Path to virtual environment
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_FLASH_ATTENTION=1
cd $SCRATCH/ai-identities/performance-evals
ollama serve &

ollama pull llama3.2:3b
source $VENV/bin/activate
export HOME=$SCRATCH	
python run_openai.py --url http://localhost:11434/v1 \
    --model llama3.2:3b \
    --category 'computer science' \
    --verbosity 0 \
    --parallel 16

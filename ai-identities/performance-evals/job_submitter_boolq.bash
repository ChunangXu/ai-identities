#!/bin/bash
# We'll call this with: bash job_submitter_boolq.sh <num_instances_ollama>
# Where num_instances is the number of Ollama servers to run, default 3 (not inclusive of python node for running python script)

NUM_INSTANCES=${1:-3}  # default to 3 here
TOTAL_NODES=$((NUM_INSTANCES + 1))

# Submitting the job here
sbatch --ntasks=$TOTAL_NODES --nodes=$TOTAL_NODES \
       --job-name=job-submitter-boolq-demo-n$NUM_INSTANCES \
       --output=boolq-test-n$NUM_INSTANCES-%j.out \
       --error=boolq-test-n$NUM_INSTANCES-%j.err \
       --cpus-per-task=80 \
       --time=02:00:00 \
       --account=def-engine14 \
       script_runner_boolq.bash $NUM_INSTANCES

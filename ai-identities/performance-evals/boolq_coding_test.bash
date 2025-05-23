#!/bin/bash
#SBATCH --ntasks=17
#SBATCH --nodes=17
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --account=def-engine14
#SBATCH --job-name=boolq-debug-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# THIS SCRIPT IS FOR RUNNING WHOLE DATASET (no test flag)
# ALSO nodes,ntasks increased to 17, time increased to 2hrs
# APART FROM THAT THE SCRIPT IS ENTIRELY THE SAME

module load NiaEnv/2019b # Default module, always there unless you force purged it before
module load python/3.11.5
module load gcc/9.4.0

# Environment setup, no need to touch anything
export VENV=$SCRATCH/ai-identities/performance-evals/venv
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_FLASH_ATTENTION=1
export HOME=$SCRATCH

#IMPORTANT: when u setup and install the ollama models, they are gonna be in your home node, not scratch
# that is models are pulled into ~/.ollama/models
# ~ represents your home directory, where you login....
# so you must have those models copied from there to your scratch directory inside this ollama_home directory
# You will likely get errors if you try to add the copy now, or even try to access the models
# within the home directory, from your compute node, as they only have access to scratch
# so you must have copied (recursively) the models with cp -r ~/.ollama/models $SCRATCH/ollama_home/
# before running this script

export OLLAMA_HOME="$SCRATCH/ollama_home"
export OLLAMA_MODELS="$SCRATCH/ollama_home/models"

# Activate virtual environment
source $VENV/bin/activate

# Change to working directory
cd $SCRATCH/ai-identities/performance-evals

# Get the node list and create array
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE_ARRAY=($NODES)
echo "Allocated nodes: ${NODE_ARRAY[@]}"

# Function to get IP address of a node
get_node_ip() {
    local node=$1
    srun --nodes=1 --nodelist=$node --ntasks=1 --cpus-per-task=1 hostname -i | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
}

# Get IP addresses for each node
IP_1=$(get_node_ip ${NODE_ARRAY[0]})
IP_2=$(get_node_ip ${NODE_ARRAY[1]})
IP_3=$(get_node_ip ${NODE_ARRAY[2]})

echo "Node IPs: $IP_1, $IP_2, $IP_3"

# Function to start Ollama server
start_ollama_server() {
    local node=$1
    local ip=$2
    local port=$3
    local cpus=$4
    local log_file="ollama_${node}_${port}.log"

    echo "Starting Ollama server on node $node (${ip}:${port})"

    # Set CPU threads for this Ollama instance
    export OLLAMA_CPU_THREADS=$cpus

    srun --nodes=1 --nodelist=$node \
        --ntasks=1 \
        --cpus-per-task=$cpus \
        --export=ALL,OLLAMA_HOST=${ip}:${port},OLLAMA_CPU_THREADS=$cpus \
        /bin/bash -c "
            echo 'Starting Ollama on \$(hostname) with OLLAMA_HOST=${ip}:${port}';
            $SCRATCH/ai-identities/ollama/bin/ollama serve
        " > $log_file 2>&1 &

    echo $! > "ollama_${port}.pid"
    sleep 5

    if ! ps -p $(cat "ollama_${port}.pid") > /dev/null; then
        echo "Failed to start Ollama server on $node"
        cat $log_file
        return 1
    fi

    return 0
}

# Start Ollama servers with <their_ip:port_number> and 20 CPUs for each
start_ollama_server ${NODE_ARRAY[0]} $IP_1 11434 20 || exit 1
start_ollama_server ${NODE_ARRAY[1]} $IP_2 11435 20 || exit 1
start_ollama_server ${NODE_ARRAY[2]} $IP_3 11436 20 || exit 1

sleep 10

echo "All Ollama servers are running. Starting evaluation..."
echo "VENV: $VENV"

# Run evaluation script on the fourth node
### THIS EXPORT IS CRUCIAL, USED IN the python eval script
### Since each ollama/openAI model is running on a different node
### and each node will have a uniqe IP address, that wont be the localhost
export OLLAMA_SERVERS="$IP_1:11434,$IP_2:11435,$IP_3:11436"

# Run the evaluation script
srun --nodes=1 --nodelist=${NODE_ARRAY[3]} --ntasks=1 --cpus-per-task=20 \
    python boolq_eval.py --model llama3.2 --parallel 16
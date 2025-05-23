#!/bin/bash
# This is script_runner_boolq.bash - the actual script that runs after sbatch

# Get number of Ollama instances from first argument
NUM_INSTANCES=${1:-3}  # Default to 3 if not provided
echo "Setting up with $NUM_INSTANCES Ollama instances"

module load NiaEnv/2019b 
module load python/3.11.5
module load gcc/9.4.0

# Environment setup
export VENV=$SCRATCH/ai-identities/performance-evals/venv
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_FLASH_ATTENTION=1
export HOME=$SCRATCH
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
echo "Number of Ollama instances: $NUM_INSTANCES"

# Function to get IP address of a node
get_node_ip() {
    local node=$1
    srun --nodes=1 --nodelist=$node --ntasks=1 --cpus-per-task=1 hostname -i | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
}

# Get IP addresses for each node and store in array
IP_ARRAY=()
for ((i=0; i<$NUM_INSTANCES; i++)); do
    IP_ARRAY[$i]=$(get_node_ip ${NODE_ARRAY[$i]})
    echo "Node ${NODE_ARRAY[$i]} IP: ${IP_ARRAY[$i]}"
done

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

# Start all Ollama servers with their respective IPs and ports
OLLAMA_SERVERS=""
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((11434 + i))
    start_ollama_server ${NODE_ARRAY[$i]} ${IP_ARRAY[$i]} $PORT 20 || exit 1
    
    # Build the OLLAMA_SERVERS string
    if [ -z "$OLLAMA_SERVERS" ]; then
        OLLAMA_SERVERS="${IP_ARRAY[$i]}:$PORT"
    else
        OLLAMA_SERVERS="$OLLAMA_SERVERS,${IP_ARRAY[$i]}:$PORT"
    fi
done

sleep 10

echo "All Ollama servers are running. Starting evaluation..."
echo "OLLAMA_SERVERS: $OLLAMA_SERVERS"
echo "VENV: $VENV"

# Export the OLLAMA_SERVERS environment variable
export OLLAMA_SERVERS="$OLLAMA_SERVERS"

# Run the evaluation script on the last node
EVAL_NODE=${NODE_ARRAY[$NUM_INSTANCES]}
echo "Running evaluation on node: $EVAL_NODE"

# Run the evaluation script
# --> DONT FORGET TO CHANGE THIS TO YOUR EVALUATION NAME
srun --nodes=1 --nodelist=$EVAL_NODE --ntasks=1 --cpus-per-task=20 \
    python boolq_eval.py --model llama3.2 --parallel 16
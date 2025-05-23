#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --cpus-per-task=80
#SBATCH --time=00:20:00
#SBATCH --account=def-engine14
#SBATCH --job-name=boolq-new-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# This is to test if your script runs with multiple ollama nodes
# Since debug queue max 4 nodes as well as time restriction of 22.5mins (for 1 node, 1 hour, for 4 nodes, 22.5mins) 
# you have 2 ways to run smth in debug queue, either
# A) run debugjob --clean 4, to request 4 nodes,
    # This will salloc (interactive) 4 nodes, and you can run this script in the interactive shell
    # by running bash debug_boolq_test_ollama.bash
    # main use, this is compute node, so if your scripts arent working, you can literally "debug" them here
    # by sending commands to see your PATH variables, your ollama models being accessible/downloaded or not
    # check your venv and python version/downloaded modules etc.
# B) Else just do sbatch -p debug debug_boolq_test_ollama.bash
    # This will submit the job to debug queue, and you can check the output and error files
    # to see if your script ran successfully or not
    # you can see the file name formats above, example: boolq-new-test-12345.out/err
    # This is useful if you are sure your script will run, and you just want to see the output
    # and error files to see if everything runs properly or not

# OPTIONAL: Load modules
# I did this before hand manually, but thats just because i was trying to figure out venv
# Keeping this causes no harm
module load NiaEnv/2019b # Default module, always there unless you force purged it before
module load python/3.11.5
module load gcc/9.4.0

# Environment setup, no need to touch anything
export VENV=$SCRATCH/ai-identities/performance-evals/venv
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_FLASH_ATTENTION=1
export HOME=$SCRATCH
export OLLAMA_HOME="$SCRATCH/ollama_home"
export OLLAMA_MODELS="$SCRATCH/ollama_home/models"

# Activate virtual environment
# IMPORTANT: This is the venv where you have installed the required modules before
# Has to be activated so that your models are accesible
# If transferred venv from windows, remember activation script becomes $VENV/Scripts/activate
# If transferred venv from linux, remember activation script becomes $VENV/bin/activate
# Rest nothing needs to be changed except your calling eval script
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

# Start Ollama servers with 20 CPUs each
start_ollama_server ${NODE_ARRAY[0]} $IP_1 11434 80 || exit 1
start_ollama_server ${NODE_ARRAY[1]} $IP_2 11435 80 || exit 1
start_ollama_server ${NODE_ARRAY[2]} $IP_3 11436 80 || exit 1

sleep 10

echo "All Ollama servers are running. Starting evaluation..."
echo "VENV: $VENV"

# Run evaluation script on the fourth node
export OLLAMA_SERVERS="$IP_1:11434,$IP_2:11435,$IP_3:11436"


# NODE_ARRAY[3], the fourth node, is the one where the python eval runs
# Can max out cpus per task to 40 or 80, since we have 4 nodes, and 80 cpus per node
# IMPORTANT: replace boolq_eval.py with your script name
# IMPORTANT: For now replace llama3.2 with your model name,
#               i'll make it a variable that you can pass through sbatch later,
#               along with number of ollama nodes
srun --nodes=1 --nodelist=${NODE_ARRAY[3]} --ntasks=1 --cpus-per-task=80 \
    python mmlu_pro_eval.py --url http://localhost:11434/v1 \
    --model llama3.2:3b \
    --category 'computer science' \
    --verbosity 0 \
    --parallel 16

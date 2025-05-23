# Multi-Node Model Evaluation Guide for Niagara

This guide should help explain all steps to run your script on the Niagara supercomp
> ❗ **IMPORTANT**
> NEW UPDATE: Since we would like to have arbitrary number of servers,
> Workflow for running evals has been changed, look at job_submitter_boolq.bash
> You just run 'bash job_submitter_boolq.bash <NUM_SERVERS>', and it calls sbatch <eval_bash> from within itself configuring number of nodes. (eval_bash example is script_runner_boolq.bash, not much changed except taking number of servers from passed argument)
> Apart from that, important changes in boolq_eval.py (that you could take in your eval file includes):
>   - Separate directories for saving results, eval_logs, and server/error logs
>   - Each entry into directory is timestamped, and classification report is saved both as timestamped csv as well as a json metric, to make future task of developing score dataset simpler
>   - Isolated logic for openAI vs ollama models, even though openAI models cant be run without internet access on compute nodes, atleast not much has to be refactored when running openAI locally
>   - (From older version of README): logic for handling multiple servers (all marked with comments starting as # --> as example)
## Setup Process

#### SUGGESTIONS:
- It is better to work with WSL rather than Windows... some common sources of error (which I faced) are env discrepancies: activations are through venv/Scripts/activate in Windows, rather than venv/bin/activate.
- If however you must work with Windows, i would advise not to copy over your venv files to niagara. Instead copy over everything in ai-identities except venv, and make it on Niagara itself using python3 -m venv venv, discussed later

### 1. Initial Setup

```bash
# Copy files to Niagara home/login node, with path to your private key
scp -r - i <path/to/private/key> ai-identities YOUR_USERNAME@niagara.scinet.utoronto.ca:~/

# Login to Niagara and copy to scratch
ssh -i <path/to/private/key> YOUR_USERNAME@niagara.scinet.utoronto.ca
cp -r ~/ai-identities $SCRATCH/
```

### 2. Install Ollama


> ❗ **IMPORTANT**
> These commands are from your ollama_setup.bash file, (except the cd command)
> You could try running these together in bash file itself with bash ollama_setup.bash
> Or just do them manually in terminal one after another
```bash
cd $SCRATCH/ai-identities
wget https://github.com/ollama/releases/download/v0.5.11/ollama-linux-amd64.tgz
<<<<<<< HEAD
tar -xvzf ollama-linux-amd64.tgzwget https://github.com/ollama/releases/download/v0.5.11/ollama-linux-amd64.tgz
=======
tar -xvzf ollama-linux-amd64.tgz
>>>>>>> e3c5d65a21e01870924d589004600d2e7d574df2
mkdir ollama
mv bin ollama
mv lib ollama
export PATH="$PATH:$SCRATCH/ai-identities/ollama/bin"
```
Also add this PATH thing in your .bashrc file manually (in home directory)
Use either nano or vim for it, cd ~ and then nano .bashrc
### 3. Environment Setup

```bash
# Load required modules
module load python/3.11.5
module load gcc/9.4.0

# Create and activate virtual environment
cd $SCRATCH/ai-identities/performance-evals

# If you didnt move over your venv... make a new venv
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

### 4. Installing libraries and pulling models
> ❗ **IMPORTANT**
> Again, these commands are from your setup.bash file
> You could try running these together in bash file itself with bash setup.bash
> Or just do them manually in terminal one after another

```bash
# Install dependencies
pip install -r requirements.txt
ollama serve &  # Start Ollama server

# Pull required models individually .... or using your bash script
ollama pull model1 model2 model3 ...etc

# Verify models are downloaded
cd ~
ls -a  # Look for .ollama directory
ls .ollama/models

# Copy models to scratch space
cd $SCRATCH
mkdir -p ollama_home
cp -r ~/.ollama/models $SCRATCH/ollama_home/
```

You need to do this because compute node wont be able to access files/models from HOME
even more so, since we do HOME = $SCRATCH

At this point, make sure that
- your venv (activated) has all modules you need for your script. For example, to check if tqdm is installed correctly in your venv
    - You can navigate to site-packages in your venv folder.
    - Run ls | grep tqdm to confirm that the tqdm package is present.
- models directory exist inside $SCRATCH/ollama_home/
    - Further you can do ls -a inside it to find blobs and manifest.
    - To see if <model_name> has been pulled/copied correctly, it should exist in manifests/registry.ollama.ai/library/<model_name>
- ~/.bashrc file has the updated path for ollama/bin and your structure should be

### Directory Structure
Your $SCRATCH directory must be looking like this at this point
![Niagara_directory_structure](https://github.com/user-attachments/assets/6559ed36-e89a-472d-8e70-f7a6dfa51621)

## Running Evaluations

### 1. Debug Run

First, test your setup with a debug run:
(Note: for debug, its better to use the format of debug bash file, without having number of nodes as argument)
```bash
cd $SCRATCH/ai-identities/performance-evals
sbatch -p debug debug_boolq.bash
```

Or for interactive debugging:

```bash
debugjob --clean 4  # Request 4 nodes
bash debug_boolq.bash
```

### 2. Full Evaluation

Once debug run meets your expectations/succeeds : run with the proper bash files
```bash
bash job_submitter_{youreval}.bash NUM_OLLAMA_SERVERS # Request NUM_OLLAMA_SERVERS + 1 nodes, remember max 20 nodes can be requested on compute partition
```

## Required Code for Multi-Server Support

Every evaluation script must include the following core logic for multi-server support
(copied from boolq_eval.py, if possible have a look to make sure why this code exists/
interacts with other components of the eval file):
NEW UPDATE: All comments starting with # --> # contain logic for multi servers. New code
isolates logic for openAI vs ollama models, so that running evals locally with openAI doesn't require refactoring.

```python
# Get server URLs from environment variable
server_list = os.environ.get('OLLAMA_SERVERS', '').split(',')

if not server_list[0]:
    server_list = ['localhost:11434', 'localhost:11435', 'localhost:11436']

SERVERS = [
    {"url": f"http://{server}/v1", "port": int(server.split(':')[1])}
    for server in server_list if server
]

# Create OpenAI clients for each server
clients = [
    OpenAI(
        base_url=server["url"],
        api_key=config["server"]["api_key"],
        timeout=config["server"]["timeout"]
    )
    for server in SERVERS
]

# Thread-safe server rotation
server_index = 0
lock = threading.Lock()

def get_next_server():
    global server_index
    with lock:
        server = server_index
        server_index = (server_index + 1) % len(SERVERS)
    return server
```

## Important Notes

1. **Model Location**: Models are initially downloaded to `~/.ollama/models` but must be copied to `$SCRATCH/ollama_home/models` for compute node access.

2. **Virtual Environment**: 
   - Windows venv: activate using `$VENV/Scripts/activate`
   - Linux venv: activate using `$VENV/bin/activate`

3. **Resource Allocation**:
   - Debug queue: Maximum 4 nodes, 22.5 minutes for 4 nodes or 1 hour for 1 node
   - Regular jobs: Configure resources in SBATCH directives

4. **OPTIONAL: Server Health Monitoring**: Include server health checks in your evaluation script, to make sure all openAI/ollama servers are active throughout job's duration (equal load balancing if every server's getting requests constantly)

```python
def check_server_health(client, server_url):
    try:
        response = client.chat.completions.create(
            model=config["server"]["model"],
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        logging.error(f"Server health check failed for {server_url}: {e}")
        return False
```

## Common Troubleshooting

1. **Logs show import not found**: Your venv isn't configured/active/correctly installed modules. try doing python --version and if it shows 2.7, it means module load python/3.11.5 didn't go through and venv is not being used (module load python/3.11.5 first, then activate venv, then install modules you need like pip install openAI/tqdm/datasets/huggingface)
2. **Resource Errors**: The SBATCH directives match the queue you are in, debug/compute
4. **Issues pulling model**: That means ollama models werent moved from home/.ollama/models to $SCRATCH/ollama_home (remember: ollama_home doesnt exist in scratch you have to make it with the mkdir command), or the model doesnt exist in the folder itself (setup.bash pull failed somehwere)

## Monitoring and Logs

- Check job status: `squeue -u $USER`
- Check when job starts: `squeue --start -j JOBID`
- Check job performance: `jobperf JOBID`
 (important- make sure all servers have ollama running after pull for example)

- Cancel a job: `scancel -i JOBID`
- View output logs: `cat job-name-id-*.out`
- View error logs: `cat job-name-id-*.err`
- Check Ollama server logs: `cat ollama_node_port.log`

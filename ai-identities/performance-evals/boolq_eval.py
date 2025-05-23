import os
import json
import psutil
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import toml
from openai import OpenAI
import logging
import sys
import datetime

# Create timestamp for file naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create necessary directories

# --> DONT FORGET TO CHANGE THIS TO YOUR EVALUATION NAME
os.makedirs("boolq_eval_logs", exist_ok=True)
os.makedirs("boolq_eval_results", exist_ok=True)
os.makedirs("server_logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # --> DONT FORGET TO CHANGE THIS TO YOUR EVALUATION NAME
        logging.FileHandler(f'boolq_eval_logs/log_{timestamp}.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

parser = argparse.ArgumentParser(
    # --> DONT FORGET TO CHANGE THIS TO YOUR EVALUATION NAME
    prog="python3 boolq_eval.py",
    description="Run Boolq test on multiple ollama models with load balancing",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=config.toml",
    default="config.toml",
)
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument("-t", "--test", action="store_true", help="Run with test dataset")
args = parser.parse_args()

logging.info(f"Starting script with arguments: {args}")

try:
    config = toml.load(open(args.config))
    logging.info("Successfully loaded config file")
except Exception as e:
    logging.error(f"Failed to load config file: {e}")
    sys.exit(1)

if args.parallel:
    config["test"]["parallel"] = args.parallel
if args.model:
    config["server"]["model"] = args.model

# Determine if we're using an OpenAI model
model_name = config["server"]["model"]
using_openai_model = model_name.startswith("openai:") or model_name.startswith("gpt-")

model_results_dir = f"boolq_eval_results/{model_name.replace(':', '_')}"
os.makedirs(model_results_dir, exist_ok=True)


# Initialize clients dictionary to store both Ollama and OpenAI clients
# IMPORTANT: separated previous logic of ollama and openai models being run on same client
# is helpful for debugging, especially when all models will be run automatically (iteratively) on multiple nodes
# instead of being each time calling the eval script with a particular model name
clients_dict = {}

# Initialize OpenAI clients if using an OpenAI model
openai_clients = []
if using_openai_model:
    try:
        openai_api_key = config["server"]["api_key"]
        if not openai_api_key:
            logging.warning("OPENAI_API_KEY environment variable not set. OpenAI models (gpt-*) will not work.")
            sys.exit(1)
        
        # Get OpenAI server URLs from environment variable
        openai_server_list = os.environ.get('OPENAI_SERVERS', '').split(',')
        logging.info(f"OPENAI_SERVERS environment variable: {os.environ.get('OPENAI_SERVERS', 'Not set')}")
        
        if not openai_server_list[0]:
            # Default to creating multiple clients with the same base URL
            # This helps with request parallelization even when using a single API endpoint
            num_clients = 3  # Default number of clients to create
            openai_server_list = ['api.openai.com'] * num_clients
            logging.warning(f"Using default OpenAI API with {num_clients} clients as OPENAI_SERVERS not set")
        
        for i, server in enumerate(openai_server_list):
            if server:  # Only include non-empty server strings
                client = OpenAI(
                    base_url=f"https://{server}/v1" if not server.startswith('http') else server,
                    api_key=openai_api_key,
                    timeout=config["server"]["timeout"]
                )
                openai_clients.append(client)
                clients_dict[f"openai_{i}"] = client
        
        logging.info(f"Successfully created {len(openai_clients)} OpenAI clients")
    except Exception as e:
        logging.error(f"Failed to create OpenAI clients: {e}")
        sys.exit(1)
else:
    # We get the Ollama server URLs from environment variable
    server_list = os.environ.get('OLLAMA_SERVERS', '').split(',')
    logging.info(f"OLLAMA_SERVERS environment variable: {os.environ.get('OLLAMA_SERVERS', 'Not set')}")

    if not server_list[0]:
        server_list = ['localhost:11434', 'localhost:11435', 'localhost:11436']
        logging.warning("Using default server list as OLLAMA_SERVERS not set")

    SERVERS = [
        {"url": f"http://{server}/v1", "port": int(server.split(':')[1])}
        for server in server_list if server  # Only include non-empty server strings
    ]
    logging.info(f"Configured Ollama servers: {SERVERS}")

    # Create OpenAI clients for each Ollama server
    ollama_clients = []
    try:
        for i, server in enumerate(SERVERS):
            client = OpenAI(
                base_url=server["url"],
                api_key=config["server"]["api_key"],
                timeout=config["server"]["timeout"]
            )
            ollama_clients.append(client)
            clients_dict[f"ollama_{i}"] = client
        logging.info("Successfully created OpenAI clients for Ollama servers")
    except Exception as e:
        logging.error(f"Failed to create OpenAI clients for Ollama servers: {e}")
        sys.exit(1)

results = []
predictions = []
ground_truth = []
idx_check = set()
server_index = 0
lock = threading.Lock()

def get_next_server():
    """
    Get the next server by cycling through every cycle
    Better/Easier to scale than checking last id of question
    Because the number of server nodes will be variable (eventually)"""
    global server_index
    with lock:
        if using_openai_model:
            server = server_index % len(openai_clients)
        else:
            server = server_index % len(ollama_clients)
        server_index = (server_index + 1)  # Increment the counter
    return server

def format_prompt(question, passage):
    """Format the prompt for the model."""
    return f"""Given the following passage and question, answer with only 'yes' or 'no'.

Passage: {passage}

Question: {question}

Answer:"""

def check_server_health(client, server_url, model=None):
    """Check if the server is responsive."""
    try:
        # Try a simple API call to check server health
        model_to_check = model or config["server"]["model"]
        
        # For OpenAI models, strip the "openai:" prefix if present
        if using_openai_model and model_to_check.startswith("openai:"):
            model_to_check = model_to_check[7:]  # Remove "openai:" prefix
            
        response = client.chat.completions.create(
            model=model_to_check,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        logging.info(f"Health check successful for {server_url} with model {model_to_check}")
        return True
    except Exception as e:
        logging.error(f"Server health check failed for {server_url}: {e}")
        return False

def clean_response(response):
    """Clean the model's response to get just yes/no."""
    response = response.lower().strip()
    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    else:
        return 'invalid'

def query_model(idx):
    """Send a query to the appropriate model server."""
    try:
        example = dataset_list[idx]
        logging.debug(f"Processing example {idx}")
    except IndexError:
        logging.error(f"Index {idx} is out of range for dataset size {len(dataset_list)}")
        return None, None

    if idx in idx_check:
        logging.warning(f"Index {idx} already processed")
        return None, None

    model_name = config["server"]["model"]
    prompt = format_prompt(example['question'], example['passage'])
    
    # Get the next available server index
    server_idx = get_next_server()
    
    # Check if this is an OpenAI model
    if using_openai_model:
        if server_idx >= len(openai_clients):
            logging.error(f"Invalid server index {server_idx} for {len(openai_clients)} OpenAI clients")
            return None, None
        
        client = openai_clients[server_idx]
        server_description = f"openai-server-{server_idx+1}"
        
        # Strip the "openai:" prefix if present
        if model_name.startswith("openai:"):
            actual_model_name = model_name[7:]  # Remove "openai:" prefix
        else:
            actual_model_name = model_name
    else:  # Ollama model
        if server_idx >= len(ollama_clients):
            logging.error(f"Invalid server index {server_idx} for {len(ollama_clients)} Ollama clients")
            return None, None
            
        client = ollama_clients[server_idx]
        server_description = f"ollama-server-{server_idx+1}"
        actual_model_name = model_name
        
        # Strip the "ollama:" prefix if present
        if actual_model_name.startswith("ollama:"):
            actual_model_name = actual_model_name[7:]  # Remove "ollama:" prefix
    
    logging.info(f"Attempting query to {server_description} for example {idx} with model {actual_model_name}")

    try:
        logging.debug(f"Sending request to {server_description} for example {idx}")
        response = client.chat.completions.create(
            model=actual_model_name,
            messages=[{"role":"user", "content":prompt}],
            temperature=config["inference"]["temperature"],
            max_tokens=config["inference"]["max_tokens"],
            top_p=config["inference"]["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            timeout=config["server"]["timeout"],
        )
        response_str = response.choices[0].message.content.strip()
        cleaned_response = clean_response(response_str)
        
        with lock:
            idx_check.add(idx)

        result = {
            'question': example['question'],
            'passage': example['passage'],
            'predicted': cleaned_response,
            'actual': 'yes' if example['answer'] else 'no',
            'correct': (cleaned_response == 'yes') == example['answer'],
            'server': server_description,
            'model': model_name,
            'raw_response': response_str
        }
        
        logging.info(f"Successfully processed example {idx} on {server_description}")
        logging.debug(f"Result: {result}")
        
        return result, cleaned_response

    except Exception as e:
        logging.error(f"Error querying {server_description} for example {idx}: {e}")
        return None, None

# Load BoolQ dataset
try:
    dataset_path = os.environ.get('SCRATCH')+'/ai-identities/performance-evals/boolq-data' if 'SCRATCH' in os.environ else './boolq-data'
    logging.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path, split="validation")
    dataset_list = list(dataset)
    if args.test:
        dataset_list = dataset_list[:10]
    logging.info(f"Successfully loaded dataset with {len(dataset_list)} examples")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    sys.exit(1)

# Verify servers are healthy before starting
healthy_servers = 0

# Check servers based on model type
if using_openai_model:
    for idx, client in enumerate(openai_clients):
        server_url = f"OpenAI API {idx+1}"
        if check_server_health(client, server_url, model_name):
            logging.info(f"{server_url} is healthy")
            healthy_servers += 1
        else:
            logging.error(f"{server_url} is not healthy")
else:
    # Check Ollama servers
    for idx, (server, client) in enumerate(zip(SERVERS, ollama_clients)):
        if check_server_health(client, server["url"]):
            healthy_servers += 1
        else:
            logging.error(f"Ollama server {idx+1} is not healthy")

if healthy_servers == 0:
    logging.error("No healthy servers available. Exiting.")
    sys.exit(1)

logging.info(f"Found {healthy_servers} healthy servers")

# Main evaluation loop
try:
    with ThreadPoolExecutor(max_workers=min(config["test"]["parallel"], len(dataset_list))) as executor:
        futures = {
            executor.submit(query_model, idx): idx
            for idx in range(len(dataset_list))
        }

        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            smoothing=0.0, 
            ascii=True,
            desc="Processing examples"
        ):
            try:
                idx = futures[future]
                result, cleaned_response = future.result()

                if result is None:
                    logging.warning(f"No result for example {idx}")
                    continue

                results.append(result)
                predictions.append(cleaned_response == 'yes')
                ground_truth.append(dataset_list[idx]['answer'])
                logging.debug(f"Successfully processed and stored result for example {idx}")
            except Exception as e:
                logging.error(f"Error processing result for example {idx}: {e}")

except Exception as e:
    logging.error(f"Error in main evaluation loop: {e}")
    sys.exit(1)

# Calculate and save metrics
if predictions and ground_truth:
    try:
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        logging.info(f"\nAccuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(ground_truth, predictions)
        logging.info(f"\nClassification Report:\n{report}")
        
        # Save classification report as its own csv file in results directory
        report_dict = classification_report(ground_truth, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{model_results_dir}/classification_report_{timestamp}.csv")
        logging.info(f"\nClassification Report saved to {model_results_dir}/classification_report_{timestamp}.csv")

        # Calculate server distribution
        server_distribution = results_df['server'].value_counts()
        logging.info(f"\nServer Distribution:\n{server_distribution}")
        
        # Save results to model-specific directory with timestamp
        output_file = f"{model_results_dir}/results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"\nResults saved to {output_file}")
        
        if 'True' in report_dict:
            # Binary classification
            class_metrics = {
                "precision": {
                    "class_true": report_dict['True']['precision'],
                    "class_false": report_dict['False']['precision']
                },
                "recall": {
                    "class_true": report_dict['True']['recall'],
                    "class_false": report_dict['False']['recall']
                },
                "f1-score": {
                    "class_true": report_dict['True']['f1-score'],
                    "class_false": report_dict['False']['f1-score']
                },
                "support": {
                    "class_true": int(report_dict['True']['support']),
                    "class_false": int(report_dict['False']['support'])
                },
                "macro_avg": report_dict['macro avg'],
                "weighted_avg": report_dict['weighted avg']
            }
        else:
            # Fallback if keys are different
            class_metrics = report_dict
        
        # Save metrics summary
        metrics_summary = {
            "timestamp": timestamp,
            "model": model_name,
            "accuracy": float(accuracy),
            "classification_report": class_metrics,
            "processed_examples": len(idx_check),
            "total_examples": len(dataset_list),
            "server_distribution": server_distribution.to_dict()
        }
        
        with open(f"{model_results_dir}/metrics_{timestamp}.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)
        
    except Exception as e:
        logging.error(f"Error calculating/saving metrics: {e}")
else:
    logging.error("No predictions or ground truth available to calculate metrics")

logging.info(f"\nProcessed {len(idx_check)} out of {len(dataset_list)} examples")
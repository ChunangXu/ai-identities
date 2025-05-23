import threading
import requests
import numpy as np
import json
import argparse
import uuid
import time
# Parse arguments
parser = argparse.ArgumentParser(description="Send prompts to an OpenAI-compatible LLM API and process responses.")
parser.add_argument("--url", type=str, required=True, help="OpenAI-compatible API endpoint")
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")
parser.add_argument("--model", type=str, required=True, help="Model to use for completion")
parser.add_argument("--num_requests", type=int, default=200, help="Number of requests to send")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for the model")
args = parser.parse_args()


test_dict = {}
lock = threading.Lock()
print(args.url+"/chat/completions")
print()
def get_response():
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": args.model,
        "messages": [
                      {"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens":70
    }
    try:
        response = requests.post((args.url)+"/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        words = response_json["choices"][0]["message"]["content"].replace(" ","").split(",")
        with lock:
            for i in words:
                    word_lower = i.lower()
                    if word_lower not in test_dict:
                        test_dict[word_lower] = 0
                    test_dict[word_lower] += 1
    except Exception as e:
        get_response()
        time.sleep(0.5)


for i in range(20):
    # Create and start threads
    threads = []
    for _ in range(200):
        thread = threading.Thread(target=get_response)
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()



# Compute statistics
if("/" in args.model):
    filename = f"2_{args.model.split('/')[1]}_results.json"
else:
    filename = f"2_{args.model}_results.json"
with open(filename, "w") as f:
    f.write(json.dumps(test_dict, indent="\t"))
print(f"Results saved to {filename}")
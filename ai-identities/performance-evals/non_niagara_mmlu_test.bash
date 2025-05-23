#!/bin/bash

# Script: non_niagara_mmlu_eval_loop.sh
# Description: This script runs the `non_niagara_mmlu_eval.py` Python script 26 times in a loop.
#              Before each iteration, the `eval_results` folder is deleted to ensure a clean start.
#              All console output is redirected to `results.txt`, except for progress updates and errors.
#              The script will terminate early if the Python program exits with an error code of 1.
# Usage: ./mistral_mmlu_eval_loop.sh <GEMINI_API_KEY>

# Exit immediately if a command exits with a non-zero status
set -e

<<<<<<< HEAD
# Check if API key is provided
if [ -z "$1" ]; then
    echo "Error: GEMINI API key is required."
    echo "Usage: $0 <GEMINI_API_KEY>"
    echo "Example: $0 your_gemini_api_key_here"
    exit 1
fi
=======

>>>>>>> 4a56c59775306210a6358efed1bcdad42a754ed7

# Store the Gemini API key from the first command-line argument
GEMINI_API_KEY=$1


# Loop 26 times
for i in {1..30}; do
    # Output progress to console
    echo "Starting iteration $i..." | tee /dev/tty

    # Delete the eval_results folder if it exists
    if [ -d "eval_results" ]; then
        echo "Deleting eval_results folder..."
        rm -rf eval_results
    fi

<<<<<<< HEAD
    # Run the Python script with the Gemini API parameters
    if ! python3 non_niagara_mmlu_eval.py --url https://generativelanguage.googleapis.com/v1beta/models/ \
        --model gemini-2.0-flash-lite \
        --category 'computer science' \
        --verbosity 0 \
        --parallel 256 \
        --api $GEMINI_API_KEY 2>&1 | tee /dev/tty; then
=======
    # Run the Python script with the Mistral API parameters
    # Replace with the desired Mistral model
    if ! python3 non_niagara_mmlu_eval.py --url https://api.deepinfra.com/v1/openai \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
        --category 'philosophy' \
        --verbosity 0 \
        --parallel 256 \
        --api '965sMLW4RZdByVitIRgJSUMnCaoBXoXb' \
        --output eval_results \
        --max_iterations 499; then
>>>>>>> 4a56c59775306210a6358efed1bcdad42a754ed7
        echo "Python script exited with an error. Terminating early." | tee /dev/tty
        exit 1
    fi

    # Output progress to console
    echo "Iteration $i completed." | tee /dev/tty
    echo "----------------------------------------" | tee /dev/tty
done

# Final message to console
echo "All 26 iterations completed. Results saved to results.txt." | tee /dev/tty

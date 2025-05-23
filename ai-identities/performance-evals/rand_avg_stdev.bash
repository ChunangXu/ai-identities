#!/bin/bash

# Script: non_niagara_mmlu_eval_loop.sh
# Description: This script runs the `non_niagara_mmlu_eval.py` Python script 26 times in a loop.
#              Before each iteration, the `eval_results` folder is deleted to ensure a clean start.
#              All console output is redirected to `results.txt`, except for progress updates and errors.
#              The script will terminate early if the Python program exits with an error code of 1.
# Usage: ./mistral_mmlu_eval_loop.sh <MISTRAL_API_KEY>

# Exit immediately if a command exits with a non-zero status

# Store the Mistral API key from the first command-line argument

    if ! python3 rand_avg_stdev.py --url  https://api.deepinfra.com/v1/openai   \
        --model Sao10K/L3.1-70B-Euryale-v2.2   \
        --prompt 'What are the 15 best words to describe the Earth? Write only those words on one line, in order from highest ranked to lowest ranked, each separated by the symbol "|".'\
        --temperature 1.0; then
        echo "Python script exited with an error. Terminating early." | tee /dev/tty
        exit 1
    fi
import json
import time
import random
import argparse
from tqdm import tqdm
import toml
from datasets import load_dataset
import random
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import send_to_model
from utils import extract_answer
from metrics import drop_metric

SUBSET = 5

def evaluate_model(model, test_cases, config):
    results = []
    for test_case in tqdm(test_cases, desc=f"Evaluating {model}"):
        prompt = construct_prompt(test_case)
        response = send_to_model(model, prompt, config)
        extracted_answer = extract_answer(response)
        possible_answers = set(test_case["ref_text"].split(" | "))
        for curr in possible_answers:
            if curr in extracted_answer:
                is_correct = True
        
        em_score, f1_score = (1.0, 1.0) if is_correct else drop_metric(extracted_answer, possible_answers)
        
        results.append({
            "test_case": test_case,
            "response": response,
            "extracted_answer": extracted_answer,
            "em_score": em_score,
            "f1_score": f1_score
        })
    return results

def construct_prompt(test_case):
    examples = load_examples(num_examples=3)
    prompt = f"""
You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.

# Examples
{examples}

# Your Task
---
Context: {test_case["context"]}
Question: {test_case["question"]}

Write a line of the form "Answer: $ANSWER" at the end of your response as you have seen in the previous examples.
Also, if there are multiple correct answers, separate them with a space and a pipe (|) symbol.
For example, if the correct answers are "3", "10" and "between 15 to 27", you should write "Answer: 3 | 10 | between 15 to 27"
If there is only one correct answer, you should write "Answer: $ANSWER" where $ANSWER is the correct answer.
So if your answer is "Micheal Smith", you should write "Answer: Micheal Smith" and if answer is "3.14", you should write "Answer: 3.14"
IMPORTANT: Remember, do not make unnecessary assumptions, the context provided almost always has everything you need to know. 
ALSO, not all questions have multiple correct answers. Some questions have only one correct answer, that can be expressed in multiple manners.
Example: If a questions asks about how many bananas, both 80 and "80 bananas" are correct answers.

"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on DROP dataset.")
    parser.add_argument("--config", default="config/config.toml", help="Path to config file.")
    args = parser.parse_args()

    # Load config
    config = toml.load(args.config)

    # Load test cases
    test_cases = load_drop_test_cases(subset=SUBSET)

    # Evaluate models
    models = config["server"]["models"]
    for model in models:
        results = evaluate_model(model, test_cases, config)
        stringID = model
        if model.startswith("ollama"):
            stringID = model.removeprefix("ollama:chat:")
        else:
            stringID = model.removeprefix("openai:")

        stringID = stringID.replace(":", "_")
        print("file path", f"results/{stringID}_results.json")
        save_results(results, f"results/{stringID}_results.json")

    # for model in ["openai:gpt-4o"]:
    #     results = evaluate_model(model, test_cases, config)
    #     #  cant make directory with colon in name
    #     stringID = model
    #     if model.startswith("ollama"):
    #         stringID = model.removeprefix("ollama:chat:")
    #     else:
    #         stringID = model.removeprefix("openai:")
    #     stringID = stringID.replace(":", "_")
    #     save_results(results, f"results/{stringID}_results.json")


def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def load_examples(train_path=None, num_examples=3):
    """Load and return a few example question-answer pairs from the training dataset."""
    # Load the training dataset
    dataset = load_dataset("ucinlp/drop")
    train_data = dataset["validation"]

    # Select num_examples random samples from the training data
    sampled_examples = random.sample(list(train_data), num_examples)

    # Format the examples as text
    formatted_examples = []
    for example in sampled_examples:
        context = example["passage"]
        question = example["question"]
        possibleAnswers = (" | ").join(example["answers_spans"]["spans"])  # Join multiple answers with " | ", if mulitple are actually present (plz work)

        formatted_examples.append(f"Context: {context}\nQuestion: {question}\nAnswer: {possibleAnswers}")

    return "\n\n".join(formatted_examples)  # Return as a formatted string

def load_drop_test_cases(subset=SUBSET):
    dataset = load_dataset("ucinlp/drop")
    train_data = dataset["train"]  # Use the training split

    # Sample the subset of the data
    # EDIT: forgot shuffling
    iterable_dataset = train_data.to_iterable_dataset(num_shards=128)
    finalTrain = iterable_dataset.shuffle(seed=42, buffer_size=100000)
    train_data = islice(finalTrain, SUBSET) # {SUBSET} examples for now
    
    test_cases = []
    for example in train_data:
    # Ensure answers_spans is a list
        entry = example["answers_spans"]

        # EDIT: earlier used only 1 of the possible answers to be the reference text
        # changed it in load_examples, copying the logic here
        # fuck safe handling

        test_cases.append({
            "context": example["passage"],
            "question": example["question"],
            "ref_text": (" | ").join(entry["spans"])  # Safe handling of multiple answers
        })

    return test_cases


if __name__ == "__main__":
    main()
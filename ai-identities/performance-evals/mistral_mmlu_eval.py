import argparse
import re
import sys

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from mistralai import Mistral
from mistralai.utils import RetryConfig, BackoffStrategy
from mistralai.models import SDKError

def load_mmlu(category):
    """Load MMLU dataset for specified category"""
    try:
        dataset = DatasetDict({
            "test": Dataset.from_parquet('./mmlu-data/test-00000-of-00001.parquet')
        })
        # Filter questions by category
        questions = [ex for ex in dataset["test"] if ex["category"] == category]
        print(f"Loaded {len(questions)} questions for category: {category}")
        return questions
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

def build_prompt(question, options):
    """Construct simple prompt with question and options"""
    prompt = f"Question: {question}\nOptions:"
    for i, opt in enumerate(options):
        prompt += f"\n{chr(65+i)}. {opt}"
    prompt += "\nAnswer:"
    return prompt

def get_answer(response):
    """Extract answer letter from model response"""
    match = re.search(r'\b([A-E])\b', response)
    return match.group(1) if match else None

def make_api_call(client, prompt, model):
    """Direct API call without retry decorator"""
    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        return response
    except SDKError as e:
        print(f"SDKError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

def evaluate_category(config):
    """Main evaluation function with multiple iterations"""
    client = Mistral(
        api_key=config.api_key,
        timeout_ms=3000,
        retry_config=RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=1000,
                max_interval=60000,
                exponent=1.5,
                max_elapsed_time=300000
            ),
            retry_connection_errors=True
        )
    )
    questions = load_mmlu(config.category)

    # Open a text file for writing results
    with open("accuracy.txt", "w", encoding="utf-8") as f:
        for iteration in range(30):
            correct_count = 0
            for q in tqdm(questions, desc=f"Iteration {iteration + 1}"):
                prompt = build_prompt(q["question"], q["options"])
                response = make_api_call(client, prompt, config.model)
                if response is None:
                    raise Exception("response is None")
                answer = get_answer(response.choices[0].message.content)
                if answer == q["answer"]:
                    correct_count += 1

            # Calculate accuracy
            accuracy = (correct_count / len(questions) * 100) if len(questions) > 0 else 0.0

            # Write only the numeric accuracy value to file
            f.write(f"{round(accuracy, 2)}\n")

    print("Accuracy results saved to accuracy.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU Evaluation for Mistral")
    parser.add_argument("--api-key", required=True, help="Mistral API key")
    parser.add_argument("--model", default="open-mistral-nemo", help="Model name")
    parser.add_argument("--category", required=True, help="Test category")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel requests (ignored)")

    args = parser.parse_args()
    evaluate_category(args)

import gzip
import json
import random
import re

# removed load_drop_test_cases function, logic handled in evaluate.py completely
def extract_answer(response):
    match = re.search(r"answer: (.*)", response.lower().strip())
    return match.group(1).strip() if match else response.strip()

def normalize(text):
    # Normalize text for comparison
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())
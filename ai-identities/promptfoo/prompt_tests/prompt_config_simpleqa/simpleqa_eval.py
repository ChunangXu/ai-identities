"""
SimpleQA: Measuring short-form factuality in large language models
"""

import random
import re
import pandas as pd
from typing import Any, Literal, Optional
from dataclasses import dataclass, field
import json
import openai
import ollama
import os

Message = dict[str, Any]
MessageList = list[Message]

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.

Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.

These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.

Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```
""".strip()

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))

class SamplerBase:
    """Base class for defining a model sampling method."""
    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError

class ModelSampler(SamplerBase):
    """A sampler that dynamically selects a model and generates responses."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, message_list: MessageList) -> str:
        if self.model_name.startswith("ollama:chat:"):
            response = ollama.chat(model=self.model_name.split(":")[-1], messages=message_list)
            return response["message"]["content"]

        elif self.model_name.startswith("openai:"):
            client = openai.OpenAI()  # Create client instance
            response = client.chat.completions.create(
                model=self.model_name.split(":")[-1],
                messages=message_list
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

@dataclass
class EvalResult:
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]

@dataclass
class SingleEvalResult:
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None

class SimpleQAEval:
    """SimpleQA evaluation class for testing model factuality."""
    def __init__(
        self,
        grader_model: SamplerBase,
        dataset: str = "promptfoo/prompt_tests/prompt_config_simpleqa/simpleqa_tests.csv",
        num_examples: int | None = None,
        n_repeats: int = 1,
    ):
        if dataset.startswith("http"):
            df = pd.read_csv(dataset)
        elif os.path.exists(dataset):
            df = pd.read_csv(dataset)
        else:
            raise ValueError(f"Dataset {dataset} not found!")

        examples = df.to_dict(orient="records")

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when num_examples=None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)

        self.examples = examples * n_repeats
        self.grader_model = grader_model

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = []

        for row in self.examples:
            prompt_messages = [{"content": row["Question"], "role": "user"}]
            response_text = sampler(prompt_messages)

            # Grade the response
            grade = self.grade_sample(row["Question"], row["Answer"], response_text)
            score = 1.0 if grade == "CORRECT" else 0.0

            results.append(
                SingleEvalResult(
                    score=score,
                    convo=prompt_messages + [{"content": response_text, "role": "assistant"}],
                )
            )

        valid_scores = [r.score for r in results if r.score is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return EvalResult(
            score=avg_score,
            metrics={"accuracy": avg_score},
            htmls=[],
            convos=[r.convo for r in results],
        )

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        """Grade a single sample using the grader model."""
        try:
            grader_prompt = GRADER_TEMPLATE.format(
                question=question,
                target=target,
                predicted_answer=predicted_answer,
            )
            response = self.grader_model([{"content": grader_prompt, "role": "user"}])
            match = re.search(r"(A|B|C)", response)
            if not match:
                print(f"Warning: Grader response '{response}' did not contain a valid grade")
                return "NOT_ATTEMPTED"
            letter = match.group(0)
            return CHOICE_LETTER_TO_STRING[letter]
        except Exception as e:
            print(f"Error during grading: {str(e)}")
            return "NOT_ATTEMPTED"

if __name__ == "__main__":
    models = [
        "ollama:chat:llama3.2",
        "ollama:chat:deepseek-r1:1.5b",
        "ollama:chat:qwen:1.8b",
        "ollama:chat:gemma2:2b",
        "ollama:chat:phi3:3.8b",
        "ollama:chat:mistral",
        "ollama:chat:wizardlm2",
        "openai:gpt-4o"
    ]

    results = {}

    for model in models:
        sampler = ModelSampler(model)
        simpleqa_eval = SimpleQAEval(grader_model=sampler)
        eval_result = simpleqa_eval(sampler)

        results[model] = {
            "pass": eval_result.score > 0.8,
            "score": eval_result.score,
            "reason": "High accuracy" if eval_result.score > 0.8 else "Low accuracy"
        }

    print(json.dumps(results, indent=4))

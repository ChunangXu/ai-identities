import random
import re
import pandas as pd
from typing import Any, Literal
from dataclasses import dataclass, field
import json
import openai
import ollama
import os

# results after running 20 prompts: https://app.promptfoo.dev/eval/f:07236ce9-6c97-4221-944a-93806f3c7651
# Need to clean the answers and then just compare with the prompt responses. 

# Define message structure
Message = dict[str, Any]  # Keys: role, content
MessageList = list[Message]

# Regex to extract final answer from response
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


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
            response = openai.ChatCompletion.create(
                model=self.model_name.split(":")[-1],
                messages=message_list
            )
            return response["choices"][0]["message"]["content"]

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


class Eval:
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError


EQUALITY_TEMPLATE = """
Look at the following two expressions and determine whether they are equivalent. Perform only trivial simplifications.

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

---

Respond with only "Yes" or "No".

    Expression 1: {expression1}
    Expression 2: {expression2}
""".strip()


def check_equality(sampler: SamplerBase, expr1: str, expr2: str) -> bool:
    """Checks if two mathematical expressions are equivalent using the model."""
    prompt = EQUALITY_TEMPLATE.format(expression1=expr1, expression2=expr2)
    response = sampler([{"content": prompt, "role": "user"}])
    return response.lower().strip() == "yes"


class MathEval(Eval):
    """Math evaluation class for testing model accuracy on math problems."""
    def __init__(
        self,
        equality_checker: SamplerBase,
        dataset: str,
        num_examples: int | None = None,
        n_repeats: int = 16,
    ):
        if dataset.startswith("http"):
            df = pd.read_csv(dataset)  # Load from OpenAI's dataset
        elif os.path.exists(dataset):
            df = pd.read_csv(dataset)  # Load local dataset
        else:
            raise ValueError(f"Dataset {dataset} not found!")

        examples = df.to_dict(orient="records")

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when num_examples=None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)

        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = []

        for row in self.examples:
            prompt_messages = [{"content": row["Question"], "role": "user"}]
            response_text = sampler(prompt_messages)

            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None

            score = 0 if extracted_answer is None else float(check_equality(self.equality_checker, row["Answer"], extracted_answer))

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

    datasets = {
        # "math_test": "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv",
        # "math_500_test": "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv",
        "math_tests.csv": "./math_tests.csv"  # Local file
    }

    results = {}

    for model in models:
        sampler = ModelSampler(model)

        for dataset_name, dataset_path in datasets.items():
            math_eval = MathEval(equality_checker=sampler, dataset=dataset_path)
            eval_result = math_eval(sampler)

            results[f"{model}_{dataset_name}"] = {
                "pass": eval_result.score > 0.8,
                "score": eval_result.score,
                "reason": "High accuracy" if eval_result.score > 0.8 else "Low accuracy"
            }

    print(json.dumps(results, indent=4))

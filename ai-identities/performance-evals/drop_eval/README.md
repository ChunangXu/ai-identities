# Model Evaluation on DROP Dataset

This project provides a framework for evaluating different models on the DROP dataset, calculating Exact Match (EM) and F1 scores for each model's responses.

Currently, the code facilitates the default usage of the following models (and provider)
- "llama3.2:3b"
- "gpt-4o"
- "deepseek-r1:1.5b"
- "qwen:1.8b"
- "gemma2:2b"
- "phi3:3.8b"
- "mistral:7b"
- "wizardlm2"

However, you can edit the config.toml file (in the section of server->models) to customize the set of models you wish to run the eval on

## How It Works

### Dataset Loading
The script loads the DROP dataset, selecting a shuffled-subset for evaluation.

### Prompt Construction
For each test case, the script constructs a prompt using examples from the dataset, providing context and questions.
(In short, we are using Few Shot Prompting for this eval, by providing some labelled examples to it)

### Model Evaluation
The constructed prompt, along with pre-defined system prompts are sent to the specific model and its output later undergoes post-processing to compare against expected answer(s).

### Metrics Calculation
The script calculates the Exact Match (EM) and F1 scores for each model's response.

### Saving results
The results are saved as JSON files, which include the test case, model response, extracted answer and associated metrics.

## Running the Script

To evaluate the models, you can run the script as follows:

```bash
python src/evaluate.py --config <path_to_config_file>
```

### Arguments
- `--config`: The path to the config.toml file that specifies which models to evaluate.

Example:
```bash
python src/evaluate.py --config config/config.toml
```

## Output

The evaluation results are saved in the `results/` directory as JSON files. The files are named according to the model being evaluated, with the following structure:

- `test_case`: The original test case (context, question, and ref_text which is the answer (obviously unknown to the model))
- `response`: The model's response message
- `extracted_answer`: Extracting the answer from the model's response, which could contain reasoning or the thought-process behind the answer
- `em_score`: Exact-Match score
- `f1_score`: F1 score

Results are stored in files named as follows: `results/{model_name}_results.json`

### Example Output

After running the script, the results for each model will be saved in the `results/` folder. For instance: `results/openai_gpt-4_results.json`

The contents of the file will look like this:

```json
[
    {
        "test_case": {
            "context": "Example context",
            "question": "Example question",
            "ref_text": "correct answer"
        },
        "response": "Model's answer",
        "extracted_answer": "Model's extracted answer",
        "em_score": 1.0,
        "f1_score": 1.0
    }
]
```

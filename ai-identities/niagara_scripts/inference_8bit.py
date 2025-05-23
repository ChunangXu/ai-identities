import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch

def main(args):
    # Configure 8-bit quantization for CPU
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # Load tokenizer and quantized model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="cpu",  # Use CPU only
        torch_dtype=torch.float16,
    )

    # Inference example
    inputs = tokenizer("The future of AI is", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0]))

    # Save outputs to results directory
    with open(f"{args.output_dir}/output.txt", "w") as f:
        f.write(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLaMA model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()
    main(args)
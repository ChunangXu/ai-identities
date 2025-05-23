from openai import OpenAI
import ollama
system_prompt = "IMPORTANT- in your output dont restate the context or the question. If you want explain your reasoning, but ultimately follow the ANSWER: $ANSWER format strictly, as seen in previous examples. " # System prompt for chat models
def send_to_model(model, prompt, config):
    if model.startswith("ollama"):
        # Use Ollama API
        client = ollama.Client(host=config["server"]["url"])
        response = client.chat(
            model=model.removeprefix("ollama:chat:"),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            # options={"temperature": config["inference"]["temperature"], "max_tokens": config["inference"]["max_tokens"], "top_p": config["inference"]["top_p"]},
            # temperature=config["inference"]["temperature"],
            # max_tokens=config["inference"]["max_tokens"],
            # top_p=config["inference"]["top_p"],
        )
        return response.message.content.strip()
    elif model.startswith("openai"):
        # Use OpenAI API
        client = OpenAI(api_key=config["server"]["api_key"])
        response = client.chat.completions.create(
            model=model.removeprefix("openai:"),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=config["inference"]["temperature"],
            max_tokens=config["inference"]["max_tokens"],
            top_p=config["inference"]["top_p"],
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unsupported model: {model}")
import os
from openai import OpenAI, OpenAIError

# 1. Get Mistral API Key from environment variable
mistral_api_key = os.environ.get("MISTRAL_API_KEY")

if not mistral_api_key:
    print("Error: MISTRAL_API_KEY environment variable not set.")
    print("Please set the variable before running the script:")
    print("  export MISTRAL_API_KEY='your_actual_mistral_api_key'")
    exit(1)

# 2. Initialize the OpenAI client, pointing it to Mistral's API endpoint
try:
    client = OpenAI(
        api_key=mistral_api_key,
        base_url="https://api.mistral.ai/v1/",  # <-- IMPORTANT: Point to Mistral API
    )

    # 3. Define the model and messages for the chat completion
    #    Use a valid Mistral model name (e.g., mistral-small-latest, mistral-large-latest)
    #    You can list available models via API: https://docs.mistral.ai/api/#operation/listModels
    model_name = "open-mistral-nemo" # Or "mistral-large-latest", "open-mistral-7b", etc.

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of API compatibility in simple terms."}
    ]

    print(f"--- Sending request to Mistral API (Model: {model_name}) ---")
    print(f"Using OpenAI client with base_url: {client.base_url}")

    # 4. Make the API call using the standard OpenAI client method
    chat_completion = client.chat.completions.create(
        model=model_name,        # <-- Use the specified Mistral model
        messages=messages,
        max_tokens=150           # Optional: Limit response length
    )

    # 5. Print the response content
    print("\n--- Response from Mistral ---")
    if chat_completion.choices:
        print(chat_completion.choices[0].message.content)
    else:
        print("No response content received.")

    print("\n--- Request successful! ---")

except OpenAIError as e:
    print(f"\n--- An API error occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Status Code: {e.status_code}" if hasattr(e, 'status_code') else "N/A")
    print(f"Message: {e}")
    # You can print e.response for more details if needed
    # print(f"Response Body: {e.response.text}")
except Exception as e:
    print(f"\n--- An unexpected error occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Message: {e}")

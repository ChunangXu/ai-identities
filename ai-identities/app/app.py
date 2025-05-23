import asyncio
import pickle
import os
import re
import time
import uuid
from collections import Counter
import logging
import sys
import numpy as np
import sklearn # Keep for loading pickle
from concurrent.futures import ThreadPoolExecutor, as_completed # Keep if needed for sync CPU tasks, but prefer asyncio.to_thread

# --- FastAPI Imports ---
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Path, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any

# --- Asyncio OpenAI Client ---
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

# --- FastAPI Setup ---
app = FastAPI(title="LLM Identifier API", version="1.0.0")

# --- Static Files and Templates ---
# Assume your static files (CSS, JS) are in a 'static' directory
# and your HTML templates are in a 'templates' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(script_dir, "static")
templates_dir = os.path.join(script_dir, "templates")

# Check if directories exist
if not os.path.isdir(static_dir):
    print(f"Warning: Static directory not found at {static_dir}. Creating it.")
    os.makedirs(static_dir, exist_ok=True)
if not os.path.isdir(templates_dir):
    print(f"Warning: Templates directory not found at {templates_dir}. Creating it.")
    os.makedirs(templates_dir, exist_ok=True)
    # Create a placeholder index.html if it doesn't exist
    index_html_path = os.path.join(templates_dir, "index.html")
    if not os.path.isfile(index_html_path):
        print(f"Creating placeholder index.html at {index_html_path}")
        with open(index_html_path, "w") as f:
            f.write("<html><head><title>LLM Identifier</title></head><body><h1>LLM Identifier (FastAPI)</h1><p>Frontend not fully implemented yet.</p></body></html>")


app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# --- Task Management Store (In-Memory - Same Warning Applies) ---
tasks: Dict[str, Dict[str, Any]] = {}
task_lock = asyncio.Lock() # Use asyncio.Lock for async safety

# --- Logging Configuration (Using standard Python logging) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Get the root logger or a specific logger
# logger = logging.getLogger(__name__) # Option 1: Specific logger
logger = logging.getLogger() # Option 2: Root logger (might capture logs from libraries)
logger.handlers.clear() # Clear existing handlers if configuring root logger
logger.addHandler(console_handler)
log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logger.setLevel(log_level)
# logger.propagate = False # Usually not needed unless configuring specific logger and avoiding root
logger.info(f"Logging configured. Level: {log_level_name}. Handler: Console.")
# --- End Logging Configuration ---


# --- Constants (Remain the same) ---
TRAINING_WORDS_LIST = ['life-filled', 'home.', 'wondrous', 'immense', 'ever-changing.', 'massive',
                       'enigmatic', 'complex.', 'finite.\n\n\n\n', 'lively',
                       "here'sadescriptionofearthusingtenadjectives", 'me', 'dynamic', 'beautiful',
                       'ecosystems', 'interconnected.', 'finite.', 'big', '10', 'nurturing', 'then',
                       '"diverse"', 'are', 'verdant', 'diverse', 'life-giving', 'lush', 'here', '8.',
                       'ten', 'and', 'powerful', 'precious.', "it's", 'mysterious', 'temperate',
                       'evolving', 'resilient', 'think', 'intricate', 'by', 'breathtaking.', 'varied',
                       'commas:', 'evolving.', 'describe', 'essential.', 'arid', 'i', 'separated',
                       'adjectives', 'orbiting', 'a', 'inhabited', '6.', 'revolving', 'nurturing.',
                       'need', 'swirling', 'home', 'life-supporting', '10.', 'bountiful', 'because',
                       'fertile', 'resilient.\n\n\n\n', 'precious.\n\n\n\n', 'should', 'old', 'hmm',
                       'watery', 'thriving', 'magnificent', 'life-sustaining', 'adjectives:', 'exactly',
                       'spherical', 'okay', 'earth', 'resilient.', 'the', 'only', 'beautiful.',
                       'turbulent', 'start', 'terrestrial', 'teeming.', 'its', 'life-giving.', 'dense',
                       'teeming', 'resourceful', 'ancient', 'round', '1.', 'using', 'about', 'rocky',
                       'comma.', 'volatile', 'brainstorming', 'habitable.', 'to', 'in', 'stunning',
                       'fascinating', 'abundant', 'habitable', 'aquatic', 'hospitable', 'volcanic',
                       'let', 'awe-inspiring', 'changing', '2.', 'landscapes', 'awe-inspiring.', 'of',
                       'magnetic', 'breathtaking', 'alive.', 'is', 'layered', 'planet', 'beautiful.\n\n\n\n',
                       'majestic.', 'alive', 'mountainous', 'active', 'enigmatic.', 'our',
                       'irreplaceable.', 'fragile', 'blue', 'mysterious.', 'each', 'huge',
                       'interconnected', 'separatedbycommas:\n\nblue', 'rugged', 'barren', 'so',
                       'atmospheric', 'mind', 'vital', 'finite', 'fragile.', 'inhabited.', 'first',
                       'wants', 'description', 'ever-changing', 'chaotic', 'blue.', 'vast', '',
                       'habitable.\n\n\n\n', 'precious', 'rotating', 'warm', 'large', 'spinning',
                       'expansive', '7.', 'solid', 'vibrant', 'green', 'wet', 'extraordinary.',
                       'user', 'complex', 'wondrous.', 'majestic', 'comes', 'unique', 'unique.',
                       'life-sustaining.', 'living']
TRAINING_WORDS_DICT = {word: idx for idx, word in enumerate(TRAINING_WORDS_LIST)}
LIST_OF_MODELS = ["chatgpt-4o-latest", "DeepSeek-R1-Distill-Llama-70B", "DeepSeek-R1-Turbo",
                  "DeepSeek-R1", "DeepSeek-V3", "gemini-1.5-flash", "gemini-2.0-flash-001",
                  "gemma-2-27b-it", "gemma-3-27b-it", "gpt-3.5-turbo", "gpt-4.5-preview",
                  "gpt-4o-mini", "gpt-4o", "Hermes-3-Llama-3.1-405B", "L3.1-70B-Euryale-v2.2",
                  "L3.3-70B-Euryale-v2.3", "Llama-3.1-Nemotron-70B-Instruct",
                  "Llama-3.2-90B-Vision-Instruct", "Llama-3.3-70B-Instruct-Turbo",
                  "Meta-Llama-3.1-70B-Instruct-Turbo", "Mistral-Nemo-Instruct-2407",
                  "Mixtral-8x7B-Instruct-v0.1", "MythoMax-L2-13b", "o1-mini",
                  "Phi-4-multimodal-instruct", "phi-4", "Qwen2.5-7B-Instruct",
                  "Sky-T1-32B-Preview", "WizardLM-2-8x22B"]

PROVIDER_BASE_URLS = {
    "openai": None,
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "mistral": "https://api.mistral.ai/v1/"
}

# --- Custom Exception (Remains the same) ---
class ProviderAPIError(Exception):
    """Custom exception for critical API errors from providers."""
    def __init__(self, message, status_code=None, provider=None, model=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.provider = provider
        self.model = model

    def __str__(self):
        return f"ProviderAPIError(provider={self.provider}, model={self.model}, status={self.status_code}): {self.message}"


# --- Load Model (Run at startup) ---
classifier = None # Initialize
def load_model():
    """Loads the pickled classifier model. (Synchronous ok for startup)"""
    global classifier # Modify global variable
    model_path = os.path.join(script_dir, "mlp_classifier.pkl")
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        logger.info("Model loaded successfully!")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Identification will fail.")
        classifier = None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        classifier = None

# Call load_model directly when the module is loaded
load_model()

# --- Feature Preparation (Remains the same) ---
def prepare_features(word_frequencies):
    """Prepares feature vector from word frequencies. (Synchronous CPU-bound)"""
    features = np.zeros((1, len(TRAINING_WORDS_LIST)))
    total_freq = sum(word_frequencies.values())
    if total_freq == 0:
        logger.warning("Empty word frequencies provided to prepare_features")
        return features
    for word, freq in word_frequencies.items():
        if word in TRAINING_WORDS_DICT:
            idx = TRAINING_WORDS_DICT[word]
            features[0, idx] = freq / total_freq
    logger.debug(f"Prepared features. Total frequency: {total_freq}")
    if np.sum(features) == 0:
        logger.warning("No overlapping words found between input and training set")
    return features

# --- Async LLM Querying (Mostly the same, using standard logger) ---
async def query_llm(api_key: str, provider: str, model: str, num_samples: int = 100, temperature: float = 0.7, progress_callback: Optional[callable] = None, task_id: Optional[str] = None):
    """
    Query the LLM asynchronously using AsyncOpenAI, configure base_url, and report progress.
    Propagates critical errors by raising ProviderAPIError.
    (Logging uses the globally configured logger)
    """
    all_responses = []
    prompt = "What are the 15 best words to describe the Earth? Write only those words on one line, in order from highest ranked to lowest ranked, each separated by the symbol \"|\"."

    provider_lower = provider.lower()
    base_url = PROVIDER_BASE_URLS.get(provider_lower)

    log_prefix = f"[Task={task_id}] " if task_id else ""
    concurrency_limit = 50
    semaphore = asyncio.Semaphore(concurrency_limit)
    logger.info(f"{log_prefix}Starting async LLM query: Provider={provider_lower}, Model={model}, Samples={num_samples}, Concurrency={concurrency_limit}, Temp={temperature}, BaseURL={base_url or 'Default OpenAI'}")

    default_headers = {}
    if provider_lower == 'anthropic':
        logger.warning(f"{log_prefix}Attempting Anthropic call via AsyncOpenAI client. Requires compatible endpoint/proxy.")
    if provider_lower == 'google':
        logger.warning(f"{log_prefix}Attempting Google call via AsyncOpenAI client. Base URL might expect different API structure/model format.")

    client = None
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            timeout=5,
            max_retries=5
        )
    except Exception as client_err:
        msg = f"Failed to initialize Async API client: {client_err}"
        logger.error(f"{log_prefix} {msg}", exc_info=False)
        raise ProviderAPIError(msg, provider=provider_lower, model=model) from client_err

    completed_count = 0
    first_critical_error = None

    async def send_single_request_worker(attempt_index):
        nonlocal first_critical_error
        if first_critical_error: return None

        async with semaphore:
            if first_critical_error: return None

            worker_log_prefix = f"{log_prefix}[Worker {attempt_index+1}] "
            try:
                request_start_time = time.monotonic()
                logger.debug(f"{worker_log_prefix}Attempting API call to {provider_lower}/{model}")
                response_obj = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=100,
                )
                request_duration = time.monotonic() - request_start_time
                logger.debug(f"{worker_log_prefix}API call completed in {request_duration:.2f}s")

                content = response_obj.choices[0].message.content if response_obj.choices and response_obj.choices[0].message else None

                if content:
                    logger.debug(f"{worker_log_prefix}Received content: '{str(content)[:50]}...'")
                    return content
                else:
                    logger.warning(f"{worker_log_prefix}{provider_lower}/{model} yielded empty content.")
                    return None

            except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                error_msg_map = { AuthenticationError: "Auth Failed", PermissionDeniedError: "Permission Denied", NotFoundError: "Model Not Found" }
                status_code_map = { AuthenticationError: 401, PermissionDeniedError: 403, NotFoundError: 404 }
                error_type = type(e)
                error_msg = f"{error_msg_map.get(error_type, 'Auth/Resource Error')} for {provider_lower}/{model}: {e}"
                status_code = status_code_map.get(error_type)
                logger.error(f"{worker_log_prefix}{error_msg}", exc_info=False)
                raise ProviderAPIError(error_msg, status_code=status_code, provider=provider_lower, model=model) from e

            except APIStatusError as e:
                status_code = e.status_code
                error_msg = f"API Status Error {status_code} for {provider_lower}/{model}"
                is_critical = False
                if status_code in [400, 422]:
                    detailed_message = f"Invalid request ({status_code}) for {provider_lower}/{model}: {str(e)}"
                    try: error_body = e.response.json(); detailed_message = f"{detailed_message} - {error_body.get('message', '')}"
                    except: pass
                    error_msg = detailed_message
                    is_critical = True
                elif status_code == 429:
                    error_msg = f"Rate Limit Exceeded (429) for {provider_lower}/{model}. Request failed."
                    logger.warning(f"{worker_log_prefix}{error_msg}", exc_info=False)
                    return None
                elif status_code >= 500:
                    error_msg = f"Server Error ({status_code}) from {provider_lower}/{model}. Request failed."
                    logger.warning(f"{worker_log_prefix}{error_msg}: {e}", exc_info=False)
                    return None
                else:
                    error_msg = f"API Client Error ({status_code}) for {provider_lower}/{model}: {e}"
                    logger.error(f"{worker_log_prefix}{error_msg}", exc_info=False)
                    is_critical = True

                if is_critical:
                    raise ProviderAPIError(error_msg, status_code=status_code, provider=provider_lower, model=model) from e
                else:
                    return None

            except APIConnectionError as e:
                logger.warning(f"{worker_log_prefix}API Connection Error for {provider_lower}/{model}: {e}. Request failed.", exc_info=False)
                return None
            except asyncio.TimeoutError:
                logger.warning(f"{worker_log_prefix}Request timed out for {provider_lower}/{model}.", exc_info=False)
                return None
            except Exception as e:
                error_msg = f"Unexpected error during API call for {provider_lower}/{model}: {type(e).__name__}"
                logger.error(f"{worker_log_prefix}{error_msg} - {e}", exc_info=True)
                raise ProviderAPIError(f"{error_msg}. See server logs.", status_code=500, provider=provider_lower, model=model) from e

    running_tasks = [
        asyncio.create_task(send_single_request_worker(i), name=f"llm-worker-{task_id}-{i}")
        for i in range(num_samples)
    ]
    processed_tasks_count = 0

    try:
        logger.info(f"{log_prefix}Processing {num_samples} tasks concurrently...")
        for future in asyncio.as_completed(running_tasks):
            processed_tasks_count += 1
            task_name = future.get_name() if hasattr(future, 'get_name') else f"Task {processed_tasks_count}"

            try:
                result = await asyncio.wait_for(future, timeout=60.0) # Keep timeout

                if result is not None:
                    all_responses.append(result)
                    completed_count += 1
                    if progress_callback:
                        try:
                            # Check if callback itself is async
                            if asyncio.iscoroutinefunction(progress_callback):
                                await progress_callback(completed_count, num_samples)
                            else:
                                # Run sync callback in thread to avoid blocking event loop if it's slow
                                await asyncio.to_thread(progress_callback, completed_count, num_samples)
                        except Exception as cb_err:
                            logger.error(f"{log_prefix}Error in progress callback: {cb_err}", exc_info=False)

            except ProviderAPIError as critical_err:
                logger.error(f"{log_prefix}Caught critical ProviderAPIError from {task_name}: {critical_err}")
                if first_critical_error is None: first_critical_error = critical_err
                logger.warning(f"{log_prefix}Attempting to cancel remaining tasks due to critical error.")
                for task in running_tasks:
                    if not task.done(): task.cancel()
                break

            except asyncio.TimeoutError:
                logger.warning(f"{log_prefix}Timeout waiting for result from {task_name} in as_completed loop.")

            except asyncio.CancelledError:
                if first_critical_error:
                    logger.warning(f"{log_prefix}{task_name} was cancelled as expected due to an earlier critical error.")
                else:
                    logger.warning(f"{log_prefix}{task_name} was cancelled unexpectedly.")

            except Exception as e:
                logger.error(f"{log_prefix}Unexpected error retrieving result from {task_name}: {type(e).__name__} - {e}", exc_info=True)
                if first_critical_error is None:
                    first_critical_error = ProviderAPIError(f"Unexpected error processing future {task_name}: {e}", status_code=500, provider=provider_lower, model=model)
                logger.warning(f"{log_prefix}Attempting to cancel remaining tasks due to unexpected error.")
                for task in running_tasks:
                    if not task.done(): task.cancel()
                break

        if first_critical_error is not None:
            logger.error(f"{log_prefix}Critical error occurred during batch API calls. Raising ProviderAPIError: {first_critical_error}")
            raise first_critical_error

        logger.info(f"{log_prefix}Finished query for {model}. Processed {processed_tasks_count} task futures, collected {len(all_responses)} non-empty responses ({completed_count} reported via callback).")
        if completed_count < num_samples and first_critical_error is None:
            logger.warning(f"{log_prefix}Collected fewer valid responses ({completed_count}) than requested ({num_samples}) likely due to non-critical errors or empty responses.")

    except ProviderAPIError:
        raise
    except Exception as e:
        logger.error(f"{log_prefix}Critical error during asyncio task management for {provider_lower}/{model}: {str(e)}", exc_info=True)
        raise ProviderAPIError(f"Asyncio task management error: {str(e)}", provider=provider_lower, model=model, status_code=500) from e
    finally:
        if client:
            await client.close()
            logger.debug(f"{log_prefix}AsyncOpenAI client closed.")
        pending = [task for task in running_tasks if not task.done()]
        if pending:
            logger.debug(f"{log_prefix}Waiting briefly for {len(pending)} potentially pending/cancelling tasks to finalize...")
            await asyncio.gather(*pending, return_exceptions=True)
            logger.debug(f"{log_prefix}Finalization wait complete.")

    return all_responses


# --- Response Processing (Remains the same, uses global logger) ---
def process_responses(responses: List[str]) -> Counter:
    """Processes raw LLM responses to extract words. (Synchronous CPU-bound)"""
    all_words = []
    processed_count = 0
    skipped_count = 0
    logger.info(f"Processing {len(responses)} raw responses...")

    for i, response in enumerate(responses):
        if not isinstance(response, str) or not response.strip():
            logger.debug(f"Skipping invalid/empty response at index {i}: Type={type(response)}, Value='{str(response)[:50]}...'")
            skipped_count += 1
            continue

        processed_count += 1
        # Extract words using the regex
        words = re.findall(r'(?:\d+\.\s*)?([A-Za-z]+)(?=\s*\||\s*$)', response)
        # Convert to lowercase and add to list
        all_words.extend(word.lower() for word in words)

    word_frequencies = Counter(all_words)
    logger.info(f"Processed {processed_count} valid responses (skipped {skipped_count}). Found {len(word_frequencies)} unique words. Total word occurrences: {len(all_words)}")
    if word_frequencies:
        top_5 = word_frequencies.most_common(5)
        logger.debug(f"Top 5 words: {top_5}")

    return word_frequencies


# --- Background Worker Function (Async - Largely the same, uses global logger) ---
async def run_identification_task(task_id: str, api_key: str, provider: str, model: str, num_samples: int, temperature: float):
    """The actual async workhorse function that runs in the background."""
    log_prefix = f"[Task={task_id}] "
    logger.info(f"{log_prefix}Async worker task started for {provider}/{model}.")

    async def update_progress(completed, total):
        """Async callback function passed to query_llm."""
        async with task_lock:
            if task_id in tasks:
                task = tasks[task_id]
                task['completed_samples'] = completed
                if task['status'] == 'pending': task['status'] = 'processing'
                # Limit logging frequency if needed
                # if completed % (total // 10 or 1) == 0 or completed == total:
                logger.debug(f"{log_prefix}Progress update: {completed}/{total} samples.")
            else:
                logger.warning(f"{log_prefix}Progress update for non-existent task ID.")

    start_run_time = time.monotonic()
    try:
        # 1. Query LLM
        logger.info(f"{log_prefix}Starting async LLM query...")
        responses = await query_llm(
            api_key, provider, model,
            num_samples, temperature,
            progress_callback=update_progress,
            task_id=task_id
        )

        async with task_lock:
            if task_id in tasks and tasks[task_id]['status'] in ['pending', 'processing']:
                tasks[task_id]['status'] = 'processing'
                tasks[task_id]['completed_samples'] = len(responses) # Final based on actual returns

        if not responses:
            raise ValueError(f"Failed to collect any valid responses from {provider}/{model}.")
        logger.info(f"{log_prefix}Collected {len(responses)} responses.")

        # --- Run Sync Operations in ThreadPool to Avoid Blocking Event Loop ---
        # FastAPI encourages using asyncio.to_thread for potentially blocking sync code
        logger.debug(f"{log_prefix}Processing responses (in thread)...")
        word_frequencies = await asyncio.to_thread(process_responses, responses)
        if not word_frequencies:
            raise ValueError("No valid words extracted from responses.")
        logger.info(f"{log_prefix}Extracted {len(word_frequencies)} unique words.")

        logger.debug(f"{log_prefix}Preparing features (in thread)...")
        features = await asyncio.to_thread(prepare_features, word_frequencies)
        if np.sum(features) == 0:
            logger.warning(f"{log_prefix}Feature vector is all zeros (no overlap with training words).")

        # 4. Predict
        if classifier is None:
            raise RuntimeError("Classifier model is not loaded.")

        logger.info(f"{log_prefix}Making prediction (in thread)...")
        # Use asyncio.to_thread for potentially CPU-intensive prediction
        raw_prediction = (await asyncio.to_thread(classifier.predict, features))[0]

        prediction = raw_prediction
        if isinstance(raw_prediction, (np.integer, np.int64)): prediction = int(raw_prediction)
        elif hasattr(raw_prediction, 'item'): prediction = raw_prediction.item()

        class_labels = []
        if hasattr(classifier, 'classes_'): class_labels = classifier.classes_.tolist()
        else: logger.warning(f"{log_prefix}Classifier missing 'classes_' attribute.")

        predicted_model = "unknown"
        predicted_index = -1
        if class_labels:
            if isinstance(prediction, str) and prediction in class_labels:
                predicted_model = prediction
                try: predicted_index = class_labels.index(prediction)
                except ValueError: predicted_index = -1
            elif isinstance(prediction, int) and 0 <= prediction < len(class_labels):
                predicted_model = class_labels[prediction]
                predicted_index = prediction
            else:
                logger.warning(f"{log_prefix}Prediction '{prediction}' (type {type(prediction)}) is invalid or out of range for class labels: {class_labels}")
        else:
            logger.error(f"{log_prefix}Cannot determine model name: class labels unavailable.")

        logger.info(f"{log_prefix}Raw prediction: {raw_prediction}, Matched Model='{predicted_model}', Index={predicted_index}")

        confidence = 0.0
        top_predictions = []
        if hasattr(classifier, 'predict_proba') and class_labels and predicted_index != -1:
            try:
                logger.debug(f"{log_prefix}Calculating probabilities (in thread)...")
                probabilities = (await asyncio.to_thread(classifier.predict_proba, features))[0]

                if len(probabilities) == len(class_labels):
                    sorted_indices = np.argsort(probabilities)[::-1]
                    top_predictions = [
                        {"model": class_labels[i], "probability": float(probabilities[i])}
                        for i in sorted_indices[:5]
                    ]
                    if 0 <= predicted_index < len(probabilities):
                        confidence = float(probabilities[predicted_index])
                    else:
                        logger.warning(f"{log_prefix} Predicted index {predicted_index} out of bounds for probabilities array len {len(probabilities)}. Confidence set to 0.")
                        confidence = 0.0

                    preds_log = [{'m': p['model'], 'p': f"{p['probability']:.4f}"} for p in top_predictions]
                    logger.info(f"{log_prefix}Confidence: {confidence:.4f}, Top 5: {preds_log}")
                else:
                    logger.error(f"{log_prefix}Probability array length ({len(probabilities)}) mismatch with class labels ({len(class_labels)})")
            except Exception as proba_error:
                logger.error(f"{log_prefix}Error getting probabilities: {proba_error}", exc_info=False)

        # 5. Prepare Result
        status_message = "success"
        final_predicted_model = predicted_model
        if predicted_model == "unknown":
            status_message = "success_unrecognized"
            final_predicted_model = "unrecognized_model"
            confidence = 0.0
            top_predictions = []
        elif np.sum(features) == 0:
            status_message = "success_no_overlap"
            logger.warning(f"{log_prefix} Setting confidence to 0 due to zero feature overlap.")
            confidence = 0.0

        result_data = {
            "provider": provider,
            "input_model": model,
            "samples_collected": len(responses),
            "unique_words_extracted": len(word_frequencies),
            "predicted_model": final_predicted_model,
            "confidence": f"{confidence:.2%}",
            "confidence_value": confidence,
            "top_predictions": top_predictions,
            "word_frequencies_top": dict(word_frequencies.most_common(20)),
            "status": status_message,
        }

        # 6. Update Task State
        async with task_lock:
            if task_id in tasks:
                task_end_time = time.monotonic()
                duration = task_end_time - tasks[task_id]['start_time_monotonic']
                tasks[task_id]['status'] = 'completed'
                tasks[task_id]['result'] = result_data
                tasks[task_id]['end_time_monotonic'] = task_end_time
                tasks[task_id]['result']['processing_time_seconds'] = round(duration, 2)
                logger.info(f"{log_prefix}Task completed successfully in {duration:.2f} seconds.")
            else:
                logger.error(f"{log_prefix}Task ID disappeared before completion update.")

    except ProviderAPIError as api_err:
        error_message = f"API Error ({api_err.status_code}): {api_err.message}"
        logger.error(f"{log_prefix}Worker failed due to ProviderAPIError: {error_message}", exc_info=False)
        async with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error_message'] = error_message
                tasks[task_id]['end_time_monotonic'] = time.monotonic()
    except ValueError as ve:
        error_message = f"Data Processing Error: {str(ve)}"
        logger.error(f"{log_prefix}Worker failed: {error_message}", exc_info=False)
        async with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error_message'] = error_message
                tasks[task_id]['end_time_monotonic'] = time.monotonic()
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(f"{log_prefix}Worker failed: {error_message}", exc_info=True)
        async with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error_message'] = error_message
                tasks[task_id]['end_time_monotonic'] = time.monotonic()

# --- Pydantic Models for Request Validation ---

class IdentifyModelRequest(BaseModel):
    api_key: str = Field(..., description="API key for the provider")
    provider: str = Field(..., description="Provider name (e.g., openai, google)")
    model: str = Field(..., description="Model identifier to query")
    num_samples: Optional[int] = Field(100, ge=10, le=4000, description="Number of samples (10-4000)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")

class TestConnectionRequest(BaseModel):
    api_key: str = Field(..., description="API key for the provider")
    provider: str = Field(..., description="Provider name (e.g., openai, google)")
    model: str = Field(..., description="Model identifier to query")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")


# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main HTML page."""
    logger.info("Serving home page")
    # Check if index.html exists
    index_html_path = os.path.join(templates_dir, "index.html")
    if not os.path.isfile(index_html_path):
        logger.error(f"index.html not found in {templates_dir}")
        raise HTTPException(status_code=500, detail="Server configuration error: index.html not found.")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/identify-model", status_code=202)
async def identify_model_start(data: IdentifyModelRequest):
    """
    Accepts model identification parameters, starts a background task,
    and returns a task ID.
    """
    start_time = time.monotonic()
    logger.info("Received request to START /api/identify-model task")

    # Pydantic handles validation based on the model definition (IdentifyModelRequest)
    # Access validated data directly:
    api_key = data.api_key
    provider = data.provider
    model = data.model
    num_samples = data.num_samples # Already validated range
    temperature = data.temperature # Already validated range

    log_api_key_snippet = f"{api_key[:4]}..." if len(api_key) > 4 else "Provided"
    logger.info(f"Identify START request params: Provider={provider}, Model={model}, Samples={num_samples}, Temp={temperature}, APIKey={log_api_key_snippet}")

    if classifier is None:
        logger.error("Classifier model not loaded, cannot start task.")
        raise HTTPException(status_code=500, detail="Server error: Classifier model not loaded.")
    if provider.lower() not in PROVIDER_BASE_URLS:
        logger.warning(f"Provider '{provider}' not explicitly listed. Attempting with default OpenAI base URL settings.")

    # Create and store task info
    task_id = str(uuid.uuid4())
    task_info = {
        "task_id": task_id,
        "status": "pending",
        "provider": provider,
        "model": model,
        "total_samples": num_samples,
        "completed_samples": 0,
        "result": None,
        "error_message": None,
        "start_time_monotonic": time.monotonic(),
        "end_time_monotonic": None
    }

    async with task_lock:
        tasks[task_id] = task_info

    logger.info(f"[Task={task_id}] Created task. Starting background async worker...")

    # Start the background task using asyncio.create_task
    # FastAPI's BackgroundTasks is typically for tasks tied to finishing the *request*.
    # For long-running independent tasks, create_task is still appropriate.
    asyncio.create_task(
        run_identification_task(task_id, api_key, provider, model, num_samples, temperature)
    )

    duration = time.monotonic() - start_time
    logger.info(f"[Task={task_id}] Identification task accepted and started in background ({duration:.3f}s). Returning task ID.")

    return {"task_id": task_id}


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str = Path(..., description="The ID of the task to check")):
    """Retrieves the status and result (if available) of a background task."""
    logger.debug(f"Request received for /api/task-status/{task_id}")

    async with task_lock:
        task_info = tasks.get(task_id) # Get a snapshot

    if not task_info:
        logger.warning(f"Task status request for unknown task_id: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    # Prepare response data based on the snapshot
    status = task_info['status']
    response_data: Dict[str, Any] = {"status": status} # Type hint for clarity

    if status == 'pending' or status == 'processing':
        response_data["completed_samples"] = task_info.get('completed_samples', 0)
        response_data["total_samples"] = task_info.get('total_samples', 1)
        logger.debug(f"Task {task_id} status: {status}, Progress: {response_data['completed_samples']}/{response_data['total_samples']}")
    elif status == 'completed':
        result_copy = task_info.get('result', {})
        # Calculate duration if needed and not present
        if 'processing_time_seconds' not in result_copy and task_info.get('end_time_monotonic') and task_info.get('start_time_monotonic'):
            duration = task_info['end_time_monotonic'] - task_info['start_time_monotonic']
            result_copy['processing_time_seconds'] = round(duration, 2)
        response_data["result"] = result_copy
        logger.debug(f"Task {task_id} status: completed.")
    elif status == 'error':
        response_data["message"] = task_info.get('error_message', 'Unknown error')
        logger.debug(f"Task {task_id} status: error, Message: {response_data['message']}")

    return JSONResponse(content=response_data) # Use JSONResponse for explicit control if needed


@app.get("/api/models")
async def get_models():
    """Returns the list of known models and supported providers."""
    logger.info("Request received for /api/models")
    known_models = LIST_OF_MODELS # Default
    if classifier and hasattr(classifier, 'classes_'):
        try:
            classifier_classes = classifier.classes_.tolist()
            if len(classifier_classes) > 0:
                known_models = sorted(list(set(classifier_classes)))
            else:
                logger.warning("classifier.classes_ is empty. Using hardcoded list.")
        except Exception as e:
            logger.error(f"Error accessing classifier.classes_: {e}. Using hardcoded list.")

    supported_providers = sorted(list(PROVIDER_BASE_URLS.keys()))
    return {
        "models": known_models,
        "supported_providers": supported_providers
    }


@app.post("/api/test-connection")
async def test_connection(data: TestConnectionRequest):
    """Tests the connection to the specified provider and model using the API key."""
    start_time = time.monotonic()
    logger.info("Received request for /api/test-connection")

    # Pydantic handles validation
    api_key = data.api_key
    provider = data.provider
    model = data.model
    temperature = data.temperature

    log_api_key_snippet = f"{api_key[:4]}..." if len(api_key) > 4 else "Provided"
    logger.info(f"Test connection params: Provider={provider}, Model={model}, Temp={temperature}, APIKey={log_api_key_snippet}")

    if provider.lower() not in PROVIDER_BASE_URLS:
        logger.warning(f"Provider '{provider}' not explicitly listed for test. Attempting with default OpenAI base URL settings.")

    try:
        logger.info(f"Attempting single async query to test connection to {provider}/{model}")
        responses = await query_llm(api_key, provider, model, num_samples=1, temperature=temperature)
        duration = time.monotonic() - start_time

        if responses and isinstance(responses[0], str) and responses[0].strip():
            logger.info(f"Test connection successful for {provider}/{model} ({duration:.2f}s).")
            return {
                "status": "success",
                "message": f"Successfully connected to {provider} and received response from {model}.",
                "response_preview": responses[0][:100] + ('...' if len(responses[0]) > 100 else ''),
                "processing_time_seconds": round(duration, 2)
            }
        elif responses:
            logger.warning(f"Test connection to {provider}/{model} returned an invalid/empty response: Type={type(responses[0])} ({duration:.2f}s)")
            # Return a 500 error as the connection worked but the response was unusable
            raise HTTPException(
                status_code=500,
                detail=f"Connected to {provider}/{model} but received an empty or invalid response."
            )
        else:
            logger.warning(f"Test connection failed: No valid response from {provider}/{model} ({duration:.2f}s). Check credentials, model, provider.")
            # Return a 500 error indicating failure to get a response
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get response from '{model}' via '{provider}'. Check credentials, model name, Base URL, and provider status."
            )

    except ProviderAPIError as api_err:
        duration = time.monotonic() - start_time
        logger.error(f"Test connection ProviderAPIError for {provider}/{model} ({duration:.2f}s): {api_err}", exc_info=False)
        status_code = api_err.status_code if isinstance(api_err.status_code, int) and 400 <= api_err.status_code < 600 else 500
        # Raise HTTPException to let FastAPI handle the response formatting
        raise HTTPException(
            status_code=status_code,
            detail=f"API Error ({api_err.status_code or 'N/A'}): {api_err.message}"
        )
    except HTTPException:
        # Re-raise HTTPException if caught (e.g., from the 'elif responses' or 'else' blocks above)
        raise
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"Test connection unexpected error for {provider}/{model} ({duration:.2f}s): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected server error occurred: {str(e)}"
        )


# --- Main Execution (Using Uvicorn) ---
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', 'localhost')
    log_level_uvicorn = log_level_name.lower() # Uvicorn uses lowercase log level names

    # Recommended way to run FastAPI apps
    logger.info(f"Starting Uvicorn server on {host}:{port}. Log level: {log_level_uvicorn}")
    uvicorn.run(
        "__main__:app", # Reference the app instance in this file
        host=host,
        port=port,
        log_level=log_level_uvicorn,
        reload=os.environ.get('FASTAPI_RELOAD', 'false').lower() == 'true', # Use reload for development if needed
        # reload_dirs=[script_dir], # Optionally specify directories to watch for reload
    )

    # Alternative: Run directly if uvicorn isn't installed/preferred (less common)
    # print("Running without Uvicorn is not standard for FastAPI.")
    # print("Install Uvicorn: pip install uvicorn[standard]")
    # print(f"To run: uvicorn {os.path.basename(__file__).replace('.py', '')}:app --host {host} --port {port} --reload")

import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import os
import glob
import pickle  # Replaced joblib with pickle

# Constants for training
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

# Create a word-to-index mapping for fast lookups and preserving order
TRAINING_WORDS_DICT = {word: idx for idx, word in enumerate(TRAINING_WORDS_LIST)}
TRAINING_WORDS_SET = set(TRAINING_WORDS_LIST)

# List of models in training data with their order
LIST_OF_MODELS = ["chatgpt-4o-latest"]

MODEL_TO_INDEX = {model: idx for idx, model in enumerate(LIST_OF_MODELS)}
MODEL_SET = set(LIST_OF_MODELS)

def load_heatmap_data(file_path):
    """Load heatmap data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            heatmap_data = json.load(f)
        return heatmap_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

def prepare_features(heatmap_data, for_training=True):
    """
    Prepare feature matrix from heatmap data
    - Creates a DataFrame with models as rows and words as columns
    - Each cell contains the normalized frequency of that word in that model
    """
    # Extract frequencies table
    normalized_frequencies = pd.DataFrame(heatmap_data['normalized_frequencies'])

    # DEBUG: Print normalized frequencies shape and sample data
    print(f"DEBUG: Initial normalized_frequencies shape: {normalized_frequencies.shape}")
    print(f"DEBUG: First 5 columns: {list(normalized_frequencies.columns)[:5]}")
    print(f"DEBUG: First 3 rows: {list(normalized_frequencies.index)[:3]}")

    # Ensure proper normalization within each model (row)
    row_sums = normalized_frequencies.sum(axis=1)
    # print(f"DEBUG: Row sums before normalization: min={row_sums.min()}, max={row_sums.max()}, mean={row_sums.mean()}")

    # Check if we have any rows with zero sum
    zero_sum_rows = row_sums[row_sums == 0].index.tolist()
    if zero_sum_rows:
        print(f"WARNING: Found {len(zero_sum_rows)} rows with zero sum. First few: {zero_sum_rows[:3]}")

    # Normalize avoiding division by zero
    normalized_data = normalized_frequencies.div(normalized_frequencies.sum(axis=1).replace(0, 1), axis=0)

    # Verify normalization worked
    new_row_sums = normalized_data.sum(axis=1)
    # print(f"DEBUG: Row sums after normalization: min={new_row_sums.min()}, max={new_row_sums.max()}, mean={new_row_sums.mean()}")

    if for_training:
        # For training data, just verify we have the expected columns
        missing_words = [word for word in TRAINING_WORDS_LIST if word not in normalized_data.columns]
        if missing_words:
            print(f"Warning: {len(missing_words)} words from TRAINING_WORDS_LIST not found in training data")
            print(f"First few missing: {missing_words[:5]}")

            # Add missing columns with zeros
            for word in missing_words:
                normalized_data[word] = 0.0

    # Display info about the normalized data
    # print(f"Feature matrix shape: {normalized_data.shape} (models × words)")

    return normalized_data

def align_validation_data(validation_data):
    """
    Align validation data with training data:
    1. Remove words in validation not in training
    2. Add words from training missing in validation with 0 values
    3. Ensure columns are in the same order as training data
    """
    # Kept only columns that exist in the training data
    valid_columns = [col for col in validation_data.columns if col in TRAINING_WORDS_SET]
    columns_to_remove = [col for col in validation_data.columns if col not in TRAINING_WORDS_SET]
    print(f"Removing {len(columns_to_remove)} words from validation data that don't exist in training")
    if columns_to_remove:
        print(f"Examples of removed words: {columns_to_remove[:5]}")

    # Created a new DataFrame with all training words
    aligned_data = pd.DataFrame(index=validation_data.index, columns=TRAINING_WORDS_LIST)
    aligned_data.fillna(0.0, inplace=True)

    # Copied values from validation data for words that exist in both
    for col in valid_columns:
        aligned_data[col] = validation_data[col]

    # DEBUG: Check aligned data before sorting
    # print(f"DEBUG: Aligned data before ensuring column order: {aligned_data.shape}")
    # print(f"DEBUG: First 5 columns: {list(aligned_data.columns)[:5]}")

    # Preserving order
    # print(f"DEBUG: Does aligned data column order match TRAINING_WORDS_LIST? {list(aligned_data.columns) == TRAINING_WORDS_LIST}")

    print(f"Validation data aligned:")
    print(f"- Original shape: {validation_data.shape}")
    print(f"- Aligned shape: {aligned_data.shape}")
    print(f"- Words removed: {len(columns_to_remove)}")
    print(f"- Words added: {len(TRAINING_WORDS_LIST) - len(valid_columns)}")

    return aligned_data

def align_model_order(data_df):
    """Align model order to match training data"""
    # DEBUG: Check input data
    # print(f"DEBUG: align_model_order input data shape: {data_df.shape}")
    # print(f"DEBUG: Input data index (first 5): {list(data_df.index)[:5]}")
    print(f"DEBUG: Number of models in MODEL_SET: {len(MODEL_SET)}")

    aligned_df = pd.DataFrame(index=LIST_OF_MODELS, columns=data_df.columns)
    aligned_df.fillna(0.0, inplace=True)

    common_models = [model for model in data_df.index if model in MODEL_SET]
    print(f"DEBUG: Number of common models found: {len(common_models)}")

    if len(common_models) == 0:
        print("ERROR: No common models found between validation data and training set!")
        print(f"DEBUG: Validation data models: {list(data_df.index)}")
        print(f"DEBUG: First few training models: {LIST_OF_MODELS[:5]}")

    for model in common_models:
        aligned_df.loc[model] = data_df.loc[model]
        # DEBUG: Verify data was copied correctly
        if not np.array_equal(aligned_df.loc[model].values, data_df.loc[model].values):
            print(f"WARNING: Data mismatch after copy for model {model}")

    print(f"Model alignment complete:")
    print(f"- Original models: {len(data_df.index)}")
    # print(f"- Common models: {len(common_models)}")
    print(f"- Final aligned models: {len(aligned_df.index)}")
    print("")
    print("")
    return aligned_df

def train_and_validate(train_file_path, validation_file_path, model_save_path="trained_model.pkl"):
    """Train on one heatmap file, validate on another, and save the model"""
    # Load training data
    train_heatmap = load_heatmap_data(train_file_path)
    if train_heatmap is None or 'normalized_frequencies' not in train_heatmap:
        print("ERROR: Training data loading or key error.")
        return None, None, None, None

    # DEBUG: Check training data structure
    # print("DEBUG: Training heatmap keys:", list(train_heatmap.keys()))

    train_data = prepare_features(train_heatmap, for_training=True)

    # Ensure training data has exactly the right columns in the right order
    if not all(word in train_data.columns for word in TRAINING_WORDS_LIST):
        print("ERROR: Not all training words are present in the training data columns!")
        missing = [word for word in TRAINING_WORDS_LIST if word not in train_data.columns]
        print(f"Missing words: {missing[:10]}...")

    train_data = train_data[TRAINING_WORDS_LIST]

    # Load validation data
    validation_heatmap = load_heatmap_data(validation_file_path)
    if validation_heatmap is None or 'normalized_frequencies' not in validation_heatmap:
        print("ERROR: Validation data loading or key error.")
        return None, None, None, None

    # DEBUG: Check validation data structure
    # print("DEBUG: Validation heatmap keys:", list(validation_heatmap.keys()))

    validation_data = prepare_features(validation_heatmap, for_training=False)

    # Align validation data words with training data
    aligned_validation = align_validation_data(validation_data)

    # Align model order to match training data
    aligned_validation = align_model_order(aligned_validation)

    # Extract numpy arrays for training and validation
    X_train = train_data.values
    y_train = train_data.index.tolist()

    X_val = aligned_validation.values
    y_val = aligned_validation.index.tolist()

    # DEBUG: Check for NaN values in training/validation data
    # print(f"DEBUG: NaN values in X_train: {np.isnan(X_train).sum()}")
    # print(f"DEBUG: NaN values in X_val: {np.isnan(X_val).sum()}")

    # Replace any NaN values with zeros
    if np.isnan(X_train).sum() > 0:
        print("WARNING: Replacing NaN values in training data with zeros")
        X_train = np.nan_to_num(X_train)

    if np.isnan(X_val).sum() > 0:
        print("WARNING: Replacing NaN values in validation data with zeros")
        X_val = np.nan_to_num(X_val)

    # Verify dimensions
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Verify matching dimensions for classification
    if X_train.shape[1] != X_val.shape[1]:
        print(f"ERROR: Feature dimension mismatch! Training: {X_train.shape[1]}, Validation: {X_val.shape[1]}")
        return None, None, None, None

    # Train the classifiers
    print("Training classifiers...")
    print("")
    print("-----------------------------------")
    print("")

    # MLP Classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',             # ReLU activation function
        solver='adam',                 # Adam optimizer
        alpha=0.0001,                  # L2 penalty parameter
        batch_size='auto',             # Automatic batch size
        learning_rate='adaptive',      # Adaptive learning rate
        max_iter=1000,                 # Maximum number of iterations
        early_stopping=False,          # Use early stopping
        validation_fraction=0.1,       # Fraction of training data for validation
        n_iter_no_change=10,           # Number of iterations with no improvement
        random_state=42                # Random state for reproducibility
    )

    # Fit the MLP classifier with try/except to catch errors
    try:
        clf.fit(X_train, y_train)
        with open(model_save_path, 'wb') as f:  # Use pickle for saving
            pickle.dump(clf, f)
        print(f"MLP model saved to {model_save_path}")
    except Exception as e:
        print(f"ERROR fitting MLP: {str(e)}")

    print("Training complete!")
    print("")
    print("-----------------------------------")
    print("")

    # Find which validation models actually have data (not all zeros)
    has_data = np.sum(X_val, axis=1) > 0
    active_models = [model for model, active in zip(y_val, has_data) if active]

    print(f"DEBUG: Number of active models in validation set: {len(active_models)}")
    print(f"DEBUG: Active models: {active_models}")

    if active_models:
        # Filter for active models
        active_indices = [i for i, active in enumerate(has_data) if active]
        X_val_active = X_val[active_indices]
        y_val_active = [y_val[i] for i in active_indices]

        y_pred = clf.predict(X_val_active)
        accuracy = accuracy_score(y_val_active, y_pred)
        print(f"\nValidation results for MLP:")
        print(f"- Active models: {len(active_models)} out of {len(y_val)}")
        print(f"- Accuracy: {accuracy:.4f}")
        print(classification_report(y_val_active, y_pred, zero_division=1))

    return clf, train_data, X_val, y_val  # Return the trained classifier


def predict_model(clf, word_frequencies, clf_name="", json_file_path=""):
    """
    Predict the model based on word frequencies

    Args:
        clf: Trained classifier
        word_frequencies: Dict of {word: frequency} for the new sample
        clf_name: Name of the classifier (for printing)
        json_file_path: Path to json file

    Returns:
        Predicted model name
    """
    # Create a features array aligned with training data
    features = np.zeros((1, len(TRAINING_WORDS_LIST)))

    # DEBUG: Print word frequencies
    # print(f"DEBUG: Input word frequencies: {len(word_frequencies)} words")
    # print(f"DEBUG: Sample words: {list(word_frequencies.items())[:5]}")

    # Normalize the input frequencies
    total_freq = sum(word_frequencies.values())
    if total_freq == 0:
        print("Warning: Empty word frequencies provided")
        return None

    # print(f"DEBUG: Total frequency: {total_freq}")

    # Fill in the features array
    words_found = 0
    for word, freq in word_frequencies.items():
        if word in TRAINING_WORDS_DICT:
            idx = TRAINING_WORDS_DICT[word]
            features[0, idx] = freq / total_freq
            words_found += 1

    # print(f"DEBUG: Found {words_found} words in training set out of {len(word_frequencies)} input words")

    # Check if we have any non-zero features
    if np.sum(features) == 0:
        print("ERROR: No overlapping words between input and training set!")
        return None

    # Make prediction
    try:
        prediction = clf.predict(features)
        print(f"DEBUG: Raw prediction: {prediction}, for model: {json_file_path}")
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        return None

    # Get probabilities if the classifier supports predict_proba
    if hasattr(clf, 'predict_proba'):
        try:
            probabilities = clf.predict_proba(features)

            # Get the top 5 most likely models
            top_indices = np.argsort(probabilities[0])[-20:][::-1]
            top_models = [clf.classes_[i] for i in top_indices]
            top_probs = [probabilities[0][i] for i in top_indices]

            k = 5
            confidence_k = top_probs[0] / np.sum(top_probs[:k])

            # Normalize the top_probs to sum to 1.0
            top_probs_normalized = [prob / sum(top_probs) for prob in top_probs]

            print(f"Predicted model using {clf_name}: {prediction[0]}, for model: {json_file_path}")
            print("")
            # Set threshold about confidence,
            # depending on disparity in probabilities between top 2 models
            lowConf = False
            if confidence_k < 0.7 and top_probs[0] / top_probs[1] < 3 and top_probs[0] - top_probs[1] < 0.3:
                lowConf = True

            if lowConf:
                print("Low confidence in prediction: Consider Manual Review")
                print(f"Confidence percentage: {confidence_k}")
                print("Consider the following results as top 5 closest predictions:")
            else:
                print("HIGH CONFIDENCE IN PREDICTION")
                print(f"Confidence percentage: {confidence_k}")
                print("Probabilities for Top 5 predictions:")

            for model, prob in zip(top_models, top_probs_normalized):
                print(f"- {model}: {prob:.4f}")
            print("")
            print("")

        except Exception as e:
            print(f"ERROR getting probabilities: {str(e)}")
    else:
        # For models without probability estimates
        print(f"Predicted model using {clf_name}: {prediction[0]}, for model: {json_file_path}")
        print("")
    print("-----------------------------------")
    print("")
    print("")
    return prediction[0]

def process_single_model_json(json_file_path):
    """
    Process a single model's word frequency JSON and return normalized frequencies

    Args:
        json_file_path: Path to the JSON file containing word frequencies

    Returns:
        Dictionary with normalized word frequencies
    """
    try:
        # Load word frequencies from JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            word_freq = json.load(f)
        print(f"Loaded {len(word_freq)} words from {json_file_path}")  # This was the missing print
        total_freq = sum(float(freq) for freq in word_freq.values())
        if total_freq == 0:
            print("WARNING: Total frequency is zero, cannot normalize")
            return {}
        normalized_freqs = {word: float(freq) / total_freq for word, freq in word_freq.items()}
        norm_sum = sum(normalized_freqs.values())
        print(f"Sum of normalized frequencies: {norm_sum:.6f} (should be very close to 1.0)")
        top_words = sorted(normalized_freqs.items(), key=lambda x: x[1], reverse=True)[:20]

        return normalized_freqs
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")
        return {}


def main():
    train_file_path = './heatmap_data_2.json'
    validation_file_path = './validation/heatmap_data_2.json'
    model_save_path = "mlp_classifier.pkl"

    # Train on one dataset and validate on another
    print("\n=== STARTING TRAINING AND VALIDATION ===\n")
    trained_clf, _, _, _ = train_and_validate(train_file_path, validation_file_path, model_save_path)

    if trained_clf is None:
        print("Failed to train classifier. Exiting.")
        return

    # Load the saved model for prediction (you can also use lst_classifiers[0] directly)
    try:
        with open(model_save_path, 'rb') as f:  # Use pickle for loading
            loaded_clf = pickle.load(f)
        print(f"\nLoaded trained MLP model from {model_save_path}")
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_save_path}")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    evaluationDirectory = "./predictionJSON_unseen"
    json_files = glob.glob(os.path.join(evaluationDirectory, "*.json"))

    print("")
    print("")
    print(f"NOW PROCESSING ALL MODELS IN DIRECTORY: {evaluationDirectory}")
    print("")
    print("---------------------------------")
    print("")
    print("")
    if not json_files:
        print("No JSON files found in the directory.")
        return

    for json_file in json_files:
        print(f"\n=== PROCESSING MODEL JSON: {json_file} ===\n") # This line was crucial
        normalized_freqs = process_single_model_json(json_file)
        if normalized_freqs:  # Check if normalized_freqs is not empty
          predict_model(loaded_clf, normalized_freqs, "MLP", json_file)
        else:
          print(f"Skipping prediction for {json_file} due to empty frequencies.")


if __name__ == "__main__":
    main()

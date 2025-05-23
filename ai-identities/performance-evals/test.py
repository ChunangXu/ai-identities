
import json
import sys
from collections import Counter
from math import sqrt

def load_json_files(file_paths):
    """
    Loads multiple JSON files and extracts word frequency data from indices 0-49.
    :param file_paths: List of file paths to JSON files.
    :return: A dictionary with indices as keys and nested dictionaries of file data as values.
    """
    word_dicts = {i: {} for i in range(50)}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k.isdigit():
                            index = int(k)
                            if 0 <= index <= 14:
                                word_dicts[index][file_path] = Counter(v)
                else:
                    print(f"Warning: {file_path} does not contain a valid dictionary structure.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return word_dicts

def find_top_words(word_dict, top_n=8):
    """
    Finds the top N most frequent words in a dictionary.
    :param word_dict: A dictionary where keys are words and values are their frequencies.
    :param top_n: Number of top frequent words to return.
    :return: A list of words sorted by frequency in descending order.
    """
    counter = Counter(word_dict)
    return [word for word, _ in sorted(counter.most_common(top_n), key=lambda x: x[1], reverse=True)]

def compare_word_lists(base_list, compare_list):
    """
    Compares two word lists based on the difference in position of words.
    :param base_list: The reference list of words.
    :param compare_list: The list to compare.
    :return: A similarity score (lower is more similar).
    """
    position_diff = 0
    
    for i, word in enumerate(base_list):
        scale = 1 - sqrt(i / 2) * 0.2
        
        if word in compare_list:
            position_diff += abs(i - compare_list.index(word)) * scale
        else:
            position_diff += abs(i - len(compare_list))
    
    return position_diff

def get_ranked_matches_per_index(base_words, comparison_data, significance_threshold=1):
    """
    Ranks all matching files for an index and determines significance.
    :param base_words: List of words from the base file.
    :param comparison_data: Dictionary mapping file paths to their word lists.
    :param significance_threshold: Minimum difference between best and second-best to be significant.
    :return: Tuple of (sorted_scores, is_significant)
    """
    # Calculate scores for all files
    scores = {}
    for file_path, words in comparison_data.items():
        scores[file_path] = compare_word_lists(base_words, words)
    
    # Sort scores from best (lowest) to worst (highest)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    
    # Check if we have enough scores to determine significance
    is_significant = False
    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_best_score = sorted_scores[1][1]
        is_significant = (second_best_score - best_score) > significance_threshold
    
    return sorted_scores, is_significant

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <base_file.json> <compare_file1.json> [compare_file2.json] ...")
        sys.exit(1)
    
    # Get all file paths from command line arguments
    file_paths = sys.argv[1:]
    base_file = file_paths[0]
    comparison_files = file_paths[1:]
    
    # Load all JSON files
    word_dicts = load_json_files(file_paths)
    
    print(f"Analyzing indices using {base_file} as the base file:")
    
    # Track file match counts and scores
    significant_match_counts = {file: 0 for file in comparison_files}
    all_match_counts = {file: 0 for file in comparison_files}
    second_place_counts = {file: 0 for file in comparison_files}
    file_scores = {file: 0 for file in comparison_files}
    significant_indices = []
    
    # For each index, find and rank matching files
    for index in sorted(word_dicts.keys()):
        available_files = list(word_dicts[index].keys())
        
        # Skip if base file doesn't have data for this index or there's nothing to compare
        if base_file not in available_files or len(available_files) < 2:
            continue
        
        base_words = find_top_words(word_dicts[index][base_file])
        
        # Prepare comparison data
        comparison_data = {
            file: find_top_words(word_dicts[index][file]) 
            for file in comparison_files 
            if file in available_files
        }
        
        if not comparison_data:
            continue
        
        # Find and rank matches for this index
        ranked_matches, is_significant = get_ranked_matches_per_index(base_words, comparison_data)
        
        # Extract best and second best matches
        best_file, best_score = ranked_matches[0]
        second_best_file = None
        second_best_score = None
        if len(ranked_matches) > 1:
            second_best_file, second_best_score = ranked_matches[1]
        
        print(f"\n--- Index {index} ---")
        print(f"Base words ({base_file}): {base_words}")
        
        for comp_file in comparison_data:
            score = compare_word_lists(base_words, comparison_data[comp_file])
            print(f"{comp_file} words: {comparison_data[comp_file]} (score: {score:.2f})")
            file_scores[comp_file] += score
        
        # Update match counts
        all_match_counts[best_file] += 1
        if second_best_file:
            second_place_counts[second_best_file] += 1
        
        # Report on this index's matches
        if is_significant:
            significant_match_counts[best_file] += 1
            significant_indices.append(index)
            print(f"Best match for index {index}: {best_file} with score {best_score:.2f} (SIGNIFICANT)")
            if second_best_file:
                print(f"Second best match: {second_best_file} with score {second_best_score:.2f}")
                print(f"Difference: {second_best_score - best_score:.2f}")
        else:
            print(f"Best match for index {index}: {best_file} with score {best_score:.2f} (not significant)")
            if second_best_file:
                print(f"Second best match: {second_best_file} with score {second_best_score:.2f}")
                print(f"Difference: {second_best_score - best_score:.2f} (below threshold of 1.1)")
    
    # Final results
    print("\n=== Final Results ===")
    
    print("\nSignificant indices:", significant_indices)
    
    print("\nSignificant match counts (higher is better):")
    for file, count in sorted(significant_match_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{file}: {count} significant index matches")
    
    print("\nFirst place match counts:")
    for file, count in sorted(all_match_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{file}: {count} index matches")
    
    print("\nSecond place match counts:")
    for file, count in sorted(second_place_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{file}: {count} second place matches")
    
    print("\nCumulative scores (lower is better):")
    for file, score in sorted(file_scores.items(), key=lambda x: x[1]):
        print(f"{file}: {score:.2f}")
    
    if significant_match_counts:
        best_significant_match = max(significant_match_counts.items(), key=lambda x: x[1])
        best_all_match = max(all_match_counts.items(), key=lambda x: x[1])
        best_score_match = min(file_scores.items(), key=lambda x: x[1])
        
        print(f"\nBest match by significant matches: {best_significant_match[0]} with {best_significant_match[1]} significant index matches")
        print(f"Best match by all matches: {best_all_match[0]} with {best_all_match[1]} index matches")
        print(f"Best match by score: {best_score_match[0]} with score {best_score_match[1]:.2f}")

if __name__ == "__main__":
    main()
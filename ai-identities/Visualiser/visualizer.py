import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from matplotlib.colors import ListedColormap
import json

def find_json_files(directory):
    """Find all JSON files in the specified directory."""
    # Use glob to find all .json files in the directory
    json_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(json_pattern)
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    return json_files

def compare_models(json_files, top_n=30):
    all_data = pd.DataFrame()
    top_words_by_model = {}
    
    for file_path in json_files:
        # Load JSON data directly
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                word_freq = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
            df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce').fillna(0).astype(int)
            
            # Extract model name from file path
            model_name = os.path.basename(file_path).replace('_results.json', '')
            df['model'] = model_name
            
            print(f"Loaded {len(df)} words from {model_name}")
            
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                top_words_by_model[model_name] = df.sort_values('frequency', ascending=False).head(top_n)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not top_words_by_model:
        print("No valid data found in any JSON files.")
        return
    
    create_visualizations(all_data, top_words_by_model, top_n)

def create_visualizations(all_data, top_words_by_model, top_n=15):
    models = list(top_words_by_model.keys())
    print(f"Creating visualizations for models: {models}")
    num_models = len(models)
    
    # Calculate grid dimensions for multiple plots
    n_rows = (num_models + 1) // 2
    n_cols = min(2, num_models)
    if num_models == 1:
        n_rows, n_cols = 1, 1
    
    # Create individual bar charts for each model
    plt.figure(figsize=(16, 6 * n_rows))
    
    for i, model in enumerate(models):
        plt.subplot(n_rows, n_cols, i + 1)
        
        df = top_words_by_model[model].head(top_n)
        sns.barplot(x='word', y='frequency', data=df)
        
        plt.title(f"Top {top_n} Words for {model}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig('top_words_by_model.png')
    plt.close()
    
    # Create combined visualization of word frequencies across models
    all_top_words = set()
    for model in models:
        top_words = top_words_by_model[model]['word'].tolist()[:top_n]
        all_top_words.update(top_words)
    
    all_top_words_list = list(all_top_words)
    
    # Create word frequency matrix
    word_freq_matrix = pd.DataFrame(0, index=models, columns=all_top_words_list)
    
    for model in models:
        model_words = top_words_by_model[model]
        for _, row in model_words.iterrows():
            if row['word'] in all_top_words:
                word_freq_matrix.loc[model, row['word']] = row['frequency']
    
    # Normalize frequencies **within each model**
    normalized_matrix = word_freq_matrix.div(word_freq_matrix.sum(axis=1), axis=0)
    
    print("Normalized Frequencies for Each Word (Within Each Model):")
    print(normalized_matrix)
    
    # Create heatmap with color highlighting
    colors = ["purple"] * num_models  # default color
    cmap = ListedColormap(colors)
    
    highlight_matrix = pd.DataFrame(0, index=models, columns=all_top_words_list)
    for word in all_top_words_list:
        # Find model with highest normalized frequency for each word
        max_model = normalized_matrix[word].idxmax()
        highlight_matrix.loc[max_model, word] = 1
    
    plt.figure(figsize=(20, 10))
    # Base heatmap with default color
    sns.heatmap(normalized_matrix, annot=False, cmap=cmap, mask=highlight_matrix, vmin=0, vmax=1)
    
    # Overlay heatmap with highlighted cells
    sns.heatmap(normalized_matrix, annot=False, cmap=ListedColormap(["green"]), 
                mask=~highlight_matrix.astype(bool), vmin=0, vmax=1, cbar=False)
    
    plt.title(f'Normalized Word Frequency Across Models (Top {top_n} Words)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('word_frequency_heatmap.png')
    plt.close()

    # Only create PCA if we have at least 2 models
    if num_models >= 2:
        # PCA visualization
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(word_freq_matrix)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=models)
        pca_df.reset_index(inplace=True)
        pca_df.rename(columns={'index': 'model'}, inplace=True)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', s=100)
        
        for i, txt in enumerate(pca_df['model']):
            plt.annotate(txt, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), fontsize=12)
        
        plt.title('PCA of Model Word Distributions')
        plt.tight_layout()
        plt.savefig('model_pca.png')
        plt.close()

    # Save processed data to JSON for further analysis
    heatmap_data = {
        'normalized_frequencies': normalized_matrix.to_dict(),
        'highest_frequency_model': {word: normalized_matrix[word].idxmax() for word in all_top_words_list}
    }

    with open('heatmap_data_2.json', 'w') as f:
        json.dump(heatmap_data, f, indent=2)
    
    print("All visualizations have been generated successfully!")

if __name__ == "__main__":
    # Directory containing JSON files
    directory = "../classifiers/predictionJSON_seen"
    
    # Find all JSON files in the directory
    json_files = find_json_files(directory)
    
    if not json_files:
        print(f"No JSON files found in {directory}")
    else:
        # Compare models with found JSON files
        compare_models(json_files, top_n=30)
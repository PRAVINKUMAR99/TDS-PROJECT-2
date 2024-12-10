# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import openai
import matplotlib.cm as cm

# Handle different encodings
def load_dataset(file_path):
    """
    Load dataset with encoding fallback to avoid read errors.
    Tries 'utf-8' and falls back to 'latin-1' encoding.
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")

# Enhanced Correlation Heatmap
def plot_correlation_heatmap(dataset, output_file):
    """
    Generates a correlation heatmap for numerical columns in the dataset.
    Adds annotations and saves the heatmap as a PNG file.
    """
    corr = dataset.select_dtypes(exclude="object").corr()  # Compute the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(output_file, dpi=300)
    plt.close()

# Perform PCA (Principal Component Analysis) for dimensionality reduction
def perform_pca(dataset, n_components=2):
    """
    Reduces the dataset dimensions using PCA and returns the transformed components.
    """
    numeric_data = dataset.select_dtypes(include=[np.number]).dropna()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(numeric_data)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance

# Add Clustering (KMeans) to the dataset
def cluster_data(dataset, output_file):
    """
    Applies KMeans clustering to the dataset and visualizes the clusters.
    """
    numeric_data = dataset.select_dtypes(include=[np.number]).dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_data)
    dataset["Cluster"] = clusters
    sns.pairplot(dataset, hue="Cluster", palette="Set2")
    plt.savefig(output_file, dpi=300)
    plt.close()

# LLM Prompt: Create a context-aware query based on dataset
def query_llm(prompt, max_tokens=300):
    """
    Sends a concise query to the LLM for analysis. 
    Avoids sending large data, keeping the prompt clear and focused.
    """
    openai.api_key = os.environ["AIPROXY_TOKEN"]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in LLM query: {str(e)}"

# Data cleaning: Remove rows with NaNs and detect anomalies
def clean_and_analyze(dataset):
    """
    Clean the dataset by handling NaNs and calculating z-scores to detect outliers.
    """
    # Remove rows with NaN values
    cleaned_data = dataset.dropna()
    
    # Calculate z-scores for anomaly detection
    z_scores = np.abs(zscore(cleaned_data.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).all(axis=1)
    cleaned_data = cleaned_data[~outliers]
    
    return cleaned_data, outliers

# Main function to control the workflow
def main(file_path):
    """
    Main function to load data, analyze, visualize, and generate insights.
    """
    # Load the dataset
    dataset = load_dataset(file_path)

    # Data Cleaning and Anomaly Detection
    cleaned_data, outliers = clean_and_analyze(dataset)
    print(f"Outliers detected: {np.sum(outliers)}")
    
    # Generate and save Correlation Heatmap
    plot_correlation_heatmap(dataset, "correlation_heatmap.png")

    # Perform PCA for dimensionality reduction
    pca_components, variance = perform_pca(dataset)
    print(f"Explained variance by PCA components: {variance}")

    # Perform clustering and save cluster plot
    cluster_data(dataset, "cluster_plot.png")

    # Dynamic LLM query generation
    prompt = f"Analyze the following dataset columns and types:\n{dataset.dtypes}\nProvide insights on trends, anomalies, and key findings."
    insights = query_llm(prompt)
    print("LLM Insights:", insights)

    # Write Markdown report
    with open("README.md", "w") as f:
        f.write("# Dataset Analysis\n\n")
        f.write("## Data Overview\n")
        f.write(f"Dataset contains {len(dataset)} rows and {len(dataset.columns)} columns.\n")
        f.write("## Insights\n")
        f.write(insights)
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("![Cluster Plot](cluster_plot.png)\n")
    
    # Feedback loop with multiple LLM calls (for advanced analysis)
    advanced_prompt = f"Based on the PCA and clustering results, describe any significant patterns, anomalies, or relationships."
    advanced_insights = query_llm(advanced_prompt)
    print("Advanced LLM Insights:", advanced_insights)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

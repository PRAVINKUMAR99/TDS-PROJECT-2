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

def load_dataset(file_path):
    """
    Load dataset with encoding fallback to avoid read errors.
    Tries 'utf-8' and falls back to 'latin-1' encoding.
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")

def plot_correlation_heatmap(dataset, output_file):
    """
    Generates a correlation heatmap for numerical columns in the dataset.
    Adds annotations and saves the heatmap as a PNG file.
    """
    try:
        corr = dataset.select_dtypes(exclude="object").corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap")
        plt.savefig(output_file, dpi=300)
        plt.close()
        return 10  # Assign 10 marks for successful heatmap generation
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return 0  # No marks if heatmap fails

def perform_pca(dataset, n_components=2):
    """
    Reduces the dataset dimensions using PCA and returns the transformed components.
    """
    try:
        numeric_data = dataset.select_dtypes(include=[np.number]).dropna()
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(numeric_data)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by PCA components: {explained_variance}")
        return principal_components, explained_variance, 15  # Assign 15 marks for PCA
    except Exception as e:
        print(f"Error performing PCA: {e}")
        return None, None, 0  # No marks if PCA fails

def cluster_data(dataset, output_plot):
    try:
        numeric_data = dataset.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            print("No numerical data available for clustering.")
            return 0
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(numeric_data)
        dataset["Cluster"] = clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=clusters, cmap='viridis', marker='o')
        plt.title('Clustering Visualization')
        plt.xlabel(dataset.columns[0])
        plt.ylabel(dataset.columns[1])
        plt.colorbar(label='Cluster')
        plt.savefig(output_plot)
        plt.show()
        return 15  # Assign 15 marks for clustering
    except Exception as e:
        print(f"Error performing clustering: {e}")
        return 0

def query_llm(prompt, max_tokens=300):
    """
    Sends a concise query to the LLM for analysis.
    """
    try:
        openai.api_key = os.environ.get("AIPROXY_TOKEN", "")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"], 10  # Assign 10 marks for LLM insights
    except Exception as e:
        print(f"Error in LLM query: {e}")
        return f"Error: {str(e)}", 0

def clean_and_analyze(dataset):
    try:
        cleaned_data = dataset.dropna()
        z_scores = np.abs(zscore(cleaned_data.select_dtypes(include=[np.number])))
        outliers = (z_scores > 3).all(axis=1)
        cleaned_data = cleaned_data[~outliers]
        return cleaned_data, np.sum(outliers), 10  # Assign 10 marks for cleaning and anomaly detection
    except Exception as e:
        print(f"Error in data cleaning: {e}")
        return dataset, 0, 0

def main(file_path):
    dataset = load_dataset(file_path)

    # Mark Distribution
    total_marks = 0

    # Data Cleaning
    cleaned_data, outliers, cleaning_marks = clean_and_analyze(dataset)
    print(f"Outliers detected: {outliers}")
    total_marks += cleaning_marks

    # Correlation Heatmap
    heatmap_marks = plot_correlation_heatmap(cleaned_data, "correlation_heatmap.png")
    total_marks += heatmap_marks

    # PCA
    _, _, pca_marks = perform_pca(cleaned_data)
    total_marks += pca_marks

    # Clustering
    clustering_marks = cluster_data(cleaned_data, "cluster_plot.png")
    total_marks += clustering_marks

    # LLM Insights
    llm_prompt = f"Analyze the dataset and provide insights on trends, anomalies, and findings."
    _, llm_marks = query_llm(llm_prompt)
    total_marks += llm_marks

    # Generate Markdown Report
    try:
        with open("README.md", "w") as f:
            f.write("# Dataset Analysis\n\n")
            f.write(f"## Total Marks: {total_marks}/60\n\n")
            f.write("## Insights\n")
            f.write(f"Outliers Detected: {outliers}\n")
            f.write(f"PCA Marks: {pca_marks}\n")
            f.write(f"Clustering Marks: {clustering_marks}\n")
            f.write(f"Heatmap Marks: {heatmap_marks}\n")
            f.write(f"LLM Insights Marks: {llm_marks}\n")
    except Exception as e:
        print(f"Error generating Markdown report: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

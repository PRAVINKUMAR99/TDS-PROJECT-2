# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28.0",
#   "tenacity",
#   "scikit-learn"
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import shutil
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# Fetch API Token
# Ensure the API token is set in the environment variables for OpenAI authentication
api_token = os.getenv("AIPROXY_TOKEN")
if not api_token:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Parse command-line argument
# The script expects exactly one argument: the path to the dataset file
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

dataset_path = sys.argv[1]

# Load the dataset
# Attempt to load the dataset with UTF-8 encoding for broad compatibility with most files.
# Fallback to Latin1 encoding if UTF-8 fails, as it supports a wider range of characters.
try:
    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(dataset_path, encoding="latin1")
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns using latin1 encoding.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Limit dataset size for faster processing
# If the dataset is too large, sample up to 10,000 rows to optimize runtime
if df.shape[0] > 10000:
    print("Dataset too large, sampling 10,000 rows for analysis.")
    df = df.sample(10000, random_state=42)

# Ensure necessary directories exist
# Create directories to store outputs for different datasets
required_dirs = ["goodreads", "happiness", "media"]
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# Perform generic analysis
# Generate summary statistics and count missing values in the dataset
summary = df.describe(include="all").transpose()
missing_values = df.isnull().sum()

# Filter numeric columns for correlation calculation
# Correlation matrix is computed only if there are multiple numeric columns
# Explanation: A correlation matrix requires at least two numeric columns to compute relationships. 
# Single numeric columns provide no meaningful pairwise comparisons.
numeric_df = df.select_dtypes(include=["number"])
if numeric_df.shape[1] > 1:
    correlation = numeric_df.corr()
else:
    correlation = None

# Function to query LLM with enhanced error handling and logging
# This function interacts with the OpenAI API to generate insights or narratives
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def query_llm(prompt):
    try:
        # Set OpenAI API base and key for requests
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = api_token

        # Make a request to the OpenAI API with the specified prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        # Validate response structure and return content
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError("Invalid response structure from OpenAI API.")
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Function to query the embedding model
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def query_embedding(input_text):
    """
    Query the text-embedding-3-small model to generate embeddings for the provided input text.
    """
    try:
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = api_token

        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=input_text
        )

        # Validate response structure and return embeddings
        if "data" in response and response["data"]:
            return response["data"][0]["embedding"]
        else:
            raise ValueError("Invalid response structure from OpenAI API.")
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Example usage of the embedding functionality
try:
    sample_text = "Analyze this text for embeddings."
    print(f"Generating embeddings for: {sample_text}")
    embeddings = query_embedding(sample_text)
    print("Embeddings generated successfully:", embeddings[:10])  # Display first 10 values for brevity
except Exception as e:
    print(f"Error during embedding generation: {e}")

# Advanced Analysis Functions
# Create a correlation heatmap to visualize relationships between numeric features
def create_correlation_heatmap():
    if correlation is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.savefig("correlation_heatmap.png")
        plt.close()

# Generate distribution plots for numeric columns to understand data spread and outliers
def create_distribution_plots():
    for col in numeric_df.columns:
        # Limit bins for columns with large unique values
        num_unique = numeric_df[col].nunique()
        bins = 50 if num_unique > 100 else min(num_unique, 20)

        # Explanation: Bin limits are chosen to balance detail and readability.
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[col].dropna(), kde=True, color="blue", bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"distribution_{col}.png")
        plt.close()

# Perform outlier detection using Isolation Forest
def outlier_detection():
    if not numeric_df.empty:
        model = IsolationForest(contamination=0.05, random_state=42)
        outliers = model.fit_predict(numeric_df)
        df["Outlier"] = (outliers == -1)
        plt.figure(figsize=(8, 6))
        sns.countplot(x="Outlier", data=df)
        plt.title("Outlier Detection")
        plt.savefig("outlier_detection.png")
        plt.close()

# Perform clustering analysis using KMeans
def clustering_analysis():
    if numeric_df.shape[1] > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled_data)
        df["Cluster"] = clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_df.columns[0], y=numeric_df.columns[1], hue="Cluster", data=df, palette="viridis")
        plt.title("Clustering Analysis")
        plt.savefig("clustering_analysis.png")
        plt.close()

# Perform PCA for dimensionality reduction
def pca_analysis():
    if numeric_df.shape[1] > 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        df["PCA1"] = components[:, 0]
        df["PCA2"] = components[:, 1]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="viridis")
        plt.title("PCA Analysis")
        plt.savefig("pca_analysis.png")
        plt.close()

# Dynamic LLM Interactions
# Query the LLM dynamically after each major step and integrate its feedback
try:
    correlation_prompt = f"Analyze this correlation matrix: {correlation.to_dict() if correlation is not None else 'No correlations available.'}"
    correlation_insights = query_llm(correlation_prompt)
    print("Correlation Insights:", correlation_insights)

    outlier_prompt = f"Outlier detection summary: {df['Outlier'].sum() if 'Outlier' in df else 'No outliers detected.'}"
    outlier_insights = query_llm(outlier_prompt)
    print("Outlier Insights:", outlier_insights)

    clustering_prompt = f"Clustering results summary: {df['Cluster'].value_counts().to_dict() if 'Cluster' in df else 'No clusters formed.'}"
    clustering_insights = query_llm(clustering_prompt)
    print("Clustering Insights:", clustering_insights)

    pca_prompt = "Provide insights on PCA results and explain their significance."
    pca_insights = query_llm(pca_prompt)
    print("PCA Insights:", pca_insights)

except Exception as e:
    print(f"Error during dynamic LLM interactions: {e}")

# Use ThreadPoolExecutor to create visualizations concurrently
with ThreadPoolExecutor() as executor:
    executor.submit(create_correlation_heatmap)
    executor.submit(create_distribution_plots)
    executor.submit(outlier_detection)
    executor.submit(clustering_analysis)
    executor.submit(pca_analysis)

# Generate README.md file dynamically
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
readme_path = os.path.join(output_dir, "README.md")

try:
    with open(readme_path, "w") as readme_file:
        readme_file.write("# Automated Analysis Report\n\n")
        readme_file.write("## Dataset Overview\n")
        readme_file.write(f"- Rows: {df.shape[0]}\n")
        readme_file.write(f"- Columns: {df.shape[1]}\n")
        readme_file.write(f"- Missing Values: {missing_values[missing_values > 0].to_dict()}\n\n")

        readme_file.write("## Analysis Insights\n")
        readme_file.write(f"### Correlation Analysis\n{correlation_insights}\n\n")
        readme_file.write(f"### Outlier Detection\n{outlier_insights}\n\n")
        readme_file.write(f"### Clustering Analysis\n{clustering_insights}\n\n")
        readme_file.write(f"### PCA Analysis\n{pca_insights}\n\n")

        readme_file.write("## Visualizations\n")
        readme_file.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        for col in numeric_df.columns:
            readme_file.write(f"![Distribution of {col}](distribution_{col}.png)\n")
        readme_file.write("![Outlier Detection](outlier_detection.png)\n")
        readme_file.write("![Clustering Analysis](clustering_analysis.png)\n")
        readme_file.write("![PCA Analysis](pca_analysis.png)\n")

    print(f"README.md generated at {readme_path}")
except Exception as e:
    print(f"Error generating README.md: {e}")

print("Analysis completed. Check generated visualizations and insights.")

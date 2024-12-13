# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai>=0.27.0",
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
        openai.api_key = api_token
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        # Validate response structure
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

# Advanced Analysis Functions
# Perform feature importance analysis
# Mutual Information for feature importance (if a target column is present)
def feature_importance_analysis(target_column):
    if target_column in df.columns and df[target_column].nunique() > 1:
        features = numeric_df.drop(columns=[target_column], errors='ignore')
        importance = mutual_info_classif(features, df[target_column], discrete_features=False)
        importance_df = pd.DataFrame({"Feature": features.columns, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
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

# Use ThreadPoolExecutor to create visualizations concurrently
with ThreadPoolExecutor() as executor:
    executor.submit(create_correlation_heatmap)
    executor.submit(create_distribution_plots)
    executor.submit(outlier_detection)
    executor.submit(clustering_analysis)
    executor.submit(pca_analysis)

# Generate narrative with robust prompt
# Create a detailed Markdown-formatted report summarizing the analysis
narrative_prompt = f"""
You are a data storytelling assistant.
Based on the following details, create a Markdown-formatted report:

- Dataset Summary: {list(df.columns)}
- Insights: {insights}
- Visualizations: ['correlation_heatmap.png', 'distribution_*.png', 'feature_importance.png', 'outlier_detection.png', 'clustering_analysis.png', 'pca_analysis.png']

Report should include:
1. **Overview of the Dataset**: Include a brief description of the dataset and its features.
2. **Key Findings from the Analysis**: Highlight major trends, patterns, and anomalies in the dataset.
3. **Visualizations**: Provide clear explanations for the visualizations created, including statistical methods and advanced analyses.
4. **Actionable Insights and Recommendations**: Suggest practical steps or decisions based on the analysis results.
5. **Summary of Data Issues**: Note any missing data, outliers, or potential quality concerns.
6. **Next Steps**: Recommend further analyses, cleaning, or data collection to improve the dataset.

Use bullet points, subheaders, and bold text where applicable to make the report structured and easy to read.
"""
try:
    story = query_llm(narrative_prompt)
except Exception as e:
    print(f"Failed to generate narrative from LLM: {e}")
    story = "Unable to generate narrative due to API issues."

# Save narrative to README.md in the appropriate directory
# The README file includes the generated narrative and links to visualizations
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w") as f:
    f.write("# Automated Analysis Report\n\n")
    f.write(story)
    f.write("\n\n## Visualizations\n")
    f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
    for col in numeric_df.columns:
        f.write(f"![Distribution of {col}](distribution_{col}.png)\n")
    f.write("![Feature Importance](feature_importance.png)\n")
    f.write("![Outlier Detection](outlier_detection.png)\n")
    f.write("![Clustering Analysis](clustering_analysis.png)\n")
    f.write("![PCA Analysis](pca_analysis.png)\n")

# Ensure all outputs are in the specified directories
# Move generated files to the output directory
shutil.move("correlation_heatmap.png", os.path.join(output_dir, "correlation_heatmap.png"))
shutil.move("feature_importance.png", os.path.join(output_dir, "feature_importance.png"))
shutil.move("outlier_detection.png", os.path.join(output_dir, "outlier_detection.png"))
shutil.move("clustering_analysis.png", os.path.join(output_dir, "clustering_analysis.png"))
shutil.move("pca_analysis.png", os.path.join(output_dir, "pca_analysis.png"))
for col in numeric_df.columns:
    shutil.move(f"distribution_{col}.png", os.path.join(output_dir, f"distribution_{col}.png"))

print(f"Analysis complete. Results saved in {output_dir}/")

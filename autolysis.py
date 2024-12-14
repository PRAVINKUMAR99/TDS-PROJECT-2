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
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# Fetch API Token
api_token = os.getenv("AIPROXY_TOKEN")
if not api_token:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Parse command-line argument
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

dataset_path = sys.argv[1]

# Load dataset
try:
    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding="latin1")
    print(f"Loaded dataset using latin1 encoding.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Limit dataset size for faster processing
if df.shape[0] > 10000:
    df = df.sample(10000, random_state=42)

# Ensure directories exist
required_dirs = ["goodreads", "happiness", "media"]
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# Summary statistics
summary = df.describe(include="all").transpose()
missing_values = df.isnull().sum()

# Numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])
correlation = numeric_df.corr() if numeric_df.shape[1] > 1 else None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def query_llm(prompt):
    try:
        openai.api_key = api_token
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a data assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying LLM: {e}")
        raise

# Visualizations
def create_correlation_heatmap():
    if correlation is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

def create_distribution_plots():
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"distribution_{col}.png")
        plt.close()

def perform_clustering():
    if numeric_df.shape[1] > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(scaled_data)
        df["Cluster"] = clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_df.columns[0], y=numeric_df.columns[1], hue="Cluster", data=df, palette="viridis")
        plt.title("Clustering")
        plt.savefig("clustering.png")
        plt.close()

# Generate narrative
narrative_prompt = f"""
The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. Columns include: {list(df.columns)}.
Key findings:
- Correlation matrix (if applicable): {correlation.idxmax().to_dict() if correlation is not None else 'No correlations found'}.
- Missing values: {missing_values[missing_values > 0].to_dict()}.
Generate a structured Markdown report including the above details and next steps.
"""
try:
    story = query_llm(narrative_prompt)
except Exception:
    story = "Unable to generate narrative due to API issues."

# Save report
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write("# Automated Analysis Report\n")
    f.write(story)

# Move visualizations
def safe_move(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)

safe_move("correlation_heatmap.png", os.path.join(output_dir, "correlation_heatmap.png"))
safe_move("clustering.png", os.path.join(output_dir, "clustering.png"))
for col in numeric_df.columns:
    safe_move(f"distribution_{col}.png", os.path.join(output_dir, f"distribution_{col}.png"))

print(f"Analysis complete. Results saved in {output_dir}/")


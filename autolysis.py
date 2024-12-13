# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "tenacity",
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

# Load the dataset
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

# Perform generic analysis
summary = df.describe(include="all").transpose()
missing_values = df.isnull().sum()
correlation = df.corr() if df.select_dtypes(include=["number"]).shape[1] > 1 else None

# Function to query LLM
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_llm(prompt):
    openai.api_key = api_token
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Generate insights
analysis_prompt = f"""
Dataset analysis summary:
- Columns: {list(df.columns)}
- Missing values: {missing_values.to_dict()}
- Correlation matrix: {correlation.to_dict() if correlation is not None else 'N/A'}

Generate insights and suggest further analyses or visualizations.
"""
insights = query_llm(analysis_prompt)

# Create visualizations
if correlation is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

# Example visualization: Distribution plot (dynamically selected column)
for col in df.select_dtypes(include=["number"]):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"distribution_{col}.png")
    plt.close()

# Generate narrative
narrative_prompt = f"""
Based on the analysis:
- Insights: {insights}
- Charts: ['correlation_heatmap.png', 'distribution_*.png']

Write a story summarizing the dataset, analysis, insights, and implications.
"""
story = query_llm(narrative_prompt)

# Save narrative to README.md
with open("README.md", "w") as f:
    f.write("# Automated Analysis Report\n\n")
    f.write(story)
    f.write("\n\n![Correlation Heatmap](correlation_heatmap.png)\n")
    for col in df.select_dtypes(include=["number"]):
        f.write(f"![Distribution of {col}](distribution_{col}.png)\n")

# Organize outputs into directories
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
shutil.move("README.md", f"{output_dir}/README.md")
shutil.move("correlation_heatmap.png", f"{output_dir}/correlation_heatmap.png")
for col in df.select_dtypes(include=["number"]):
    shutil.move(f"distribution_{col}.png", f"{output_dir}/distribution_{col}.png")

print(f"Analysis complete. Results saved in {output_dir}/")

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

# Filter numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=["number"])
if numeric_df.shape[1] > 1:
    correlation = numeric_df.corr()
else:
    correlation = None

# Function to query LLM
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_llm(prompt):
    openai.api_key = api_token
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for data analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Generate robust analysis prompt
analysis_prompt = f"""
You are assisting with data analysis.
The dataset summary is as follows:
- Columns: {list(df.columns)}
- Missing values: {missing_values.to_dict()}
- Correlation matrix: {correlation.to_dict() if correlation is not None else 'N/A'}

Task:
1. Identify key trends or anomalies in the dataset.
2. Propose relevant visualizations to uncover insights.
3. Suggest further analyses or preprocessing steps to improve data quality.

Provide your suggestions in bullet points.
"""
insights = query_llm(analysis_prompt)

# Create visualizations
if correlation is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.savefig("correlation_heatmap.png")
    plt.close()

# Example visualization: Distribution plot (dynamically selected column)
for col in numeric_df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(numeric_df[col].dropna(), kde=True, color="blue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"distribution_{col}.png")
    plt.close()

# Generate narrative with robust prompt
narrative_prompt = f"""
You are a data storytelling assistant.
Based on the following details, create a Markdown-formatted report:

- Dataset Summary: {list(df.columns)}
- Insights: {insights}
- Visualizations: ['correlation_heatmap.png', 'distribution_*.png']

Report should include:
1. An overview of the dataset.
2. Key findings from the analysis.
3. Explanations for the visualizations.
4. Actionable insights and recommendations.

Use bullet points where applicable and ensure the report is concise and insightful.
"""
story = query_llm(narrative_prompt)

# Save narrative to README.md
with open("README.md", "w") as f:
    f.write("# Automated Analysis Report\n\n")
    f.write(story)
    f.write("\n\n![Correlation Heatmap](correlation_heatmap.png)\n")
    for col in numeric_df.columns:
        f.write(f"![Distribution of {col}](distribution_{col}.png)\n")

# Organize outputs into directories
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
shutil.move("README.md", f"{output_dir}/README.md")
shutil.move("correlation_heatmap.png", f"{output_dir}/correlation_heatmap.png")
for col in numeric_df.columns:
    shutil.move(f"distribution_{col}.png", f"{output_dir}/distribution_{col}.png")

print(f"Analysis complete. Results saved in {output_dir}/")

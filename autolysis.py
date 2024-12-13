# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai>=0.27.0",
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

# Limit dataset size for faster processing
if df.shape[0] > 10000:
    print("Dataset too large, sampling 10,000 rows for analysis.")
    df = df.sample(10000, random_state=42)

# Ensure necessary directories exist
required_dirs = ["goodreads", "happiness", "media"]
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# Perform generic analysis
summary = df.describe(include="all").transpose()
missing_values = df.isnull().sum()

# Filter numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=["number"])
if numeric_df.shape[1] > 1:
    correlation = numeric_df.corr()
else:
    correlation = None

# Function to query LLM with enhanced error handling and logging
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
try:
    insights = query_llm(analysis_prompt)
except Exception as e:
    print(f"Failed to get insights from LLM: {e}")
    insights = "No insights available due to API issues."

# Concurrently create visualizations
def create_correlation_heatmap():
    if correlation is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.savefig("correlation_heatmap.png")
        plt.close()

def create_distribution_plots():
    for col in numeric_df.columns:
        # Limit bins for columns with large unique values
        num_unique = numeric_df[col].nunique()
        bins = 50 if num_unique > 100 else min(num_unique, 20)

        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[col].dropna(), kde=True, color="blue", bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"distribution_{col}.png")
        plt.close()

with ThreadPoolExecutor() as executor:
    executor.submit(create_correlation_heatmap)
    executor.submit(create_distribution_plots)

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
5. Summary of any identified data issues.
6. Next steps for further analysis or preprocessing.

Use bullet points where applicable and ensure the report is concise and insightful.
"""
try:
    story = query_llm(narrative_prompt)
except Exception as e:
    print(f"Failed to generate narrative from LLM: {e}")
    story = "Unable to generate narrative due to API issues."

# Save narrative to README.md in the appropriate directory
output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
os.makedirs(output_dir, exist_ok=True)
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w") as f:
    f.write("# Automated Analysis Report\n\n")
    f.write(story)
    f.write("\n\n![Correlation Heatmap](correlation_heatmap.png)\n")
    for col in numeric_df.columns:
        f.write(f"![Distribution of {col}](distribution_{col}.png)\n")

# Ensure all outputs are in the specified directories
shutil.move("correlation_heatmap.png", os.path.join(output_dir, "correlation_heatmap.png"))
for col in numeric_df.columns:
    shutil.move(f"distribution_{col}.png", os.path.join(output_dir, f"distribution_{col}.png"))

print(f"Analysis complete. Results saved in {output_dir}/")

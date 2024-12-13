# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "rich",
#   "scikit-learn",
#   "tenacity",
#   "ipykernel"
# ]
# ///

import os
import sys
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from rich.console import Console
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

# Initialize console for rich logging
console = Console()

# Configure logging for tenacity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.environ.get("AIPROXY_TOKEN")

# Retry settings
def retry_settings_with_logging():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )

@retry_settings_with_logging()
def load_dataset(filename):
    """Load dataset with flexible options."""
    try:
        console.log(f"[bold blue]Loading dataset:[/] {filename}")
        return pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding="ISO-8859-1")
    except Exception as e:
        console.log(f"[yellow]Fallback to alternative delimiters:[/] {e}")
        return pd.read_csv(filename, delimiter=';', encoding="utf-8")

@retry_settings_with_logging()
def encode_image(filepath):
    """Encode an image to base64 for LLM integration."""
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        console.log(f"[red]File not found: {filepath}")
        return ""

@retry_settings_with_logging()
def request_llm_insights(summary):
    """Request insights from LLM based on summary statistics."""
    console.log("[cyan]Requesting insights from LLM...")
    models = ["gpt-4o-mini"]
    for model in models:
        try:
            llm_response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant."},
                    {"role": "user", "content": f"Here is the dataset overview: {summary}. Suggest initial analyses."}
                ],
                timeout=30  # Explicit timeout for API call
            )
            return llm_response.choices[0].message['content']
        except Exception as e:
            console.log(f"[red]Model {model} failed: {e}. Retrying with the next model...")
            continue
    raise RuntimeError("All models failed to generate insights.")

@retry_settings_with_logging()
def request_visual_insights(image_data, description):
    """Request LLM to interpret visualizations."""
    if not image_data:
        console.log(f"[yellow]Skipping visualization insights for {description}.")
        return "No insights available."

    console.log("[cyan]Requesting visualization insights from LLM...")
    models = ["gpt-4o-mini"]
    for model in models:
        try:
            llm_response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data visualization analyst."},
                    {"role": "user", "content": f"Here is an image of {description}. Analyze its insights."},
                    {"role": "user", "content": image_data}
                ],
                timeout=30  # Explicit timeout for API call
            )
            return llm_response.choices[0].message['content']
        except Exception as e:
            console.log(f"[red]Visualization insights request failed: {e}")
            continue
    raise RuntimeError("All models failed to generate visualization insights.")

@retry_settings_with_logging()
def request_story_generation(summary, insights, visual_insights):
    """Generate a Markdown story with LLM."""
    console.log("[cyan]Requesting story generation from LLM...")
    story_prompt = (
        f"Using the analysis and visualizations, generate a Markdown report. "
        f"Include dataset summary, analyses, insights, and implications. Dataset overview: {summary}. "
        f"Insights: {insights}. Visualization Insights: {visual_insights}."
    )

    models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    for model in models:
        try:
            story_response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data storytelling assistant."},
                    {"role": "user", "content": story_prompt}
                ],
                timeout=30  # Explicit timeout for API call
            )
            return story_response.choices[0].message['content']
        except Exception as e:
            console.log(f"[red]Model {model} failed: {e}. Retrying with the next model...")
            continue
    raise RuntimeError("All models failed to generate the story.")

def clean_data(data):
    """Handle missing or invalid data."""
    console.log("[cyan]Cleaning data...")
    data = data.drop_duplicates()
    data = data.dropna(how='all')
    data.fillna(data.median(numeric_only=True), inplace=True)
    return data

def detect_outliers(data):
    """Detect outliers using Isolation Forest."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.empty:
        console.log("[yellow]No numeric data found for outlier detection.")
        return data

    console.log("[cyan]Performing outlier detection...")
    model = IsolationForest(contamination=0.05, random_state=42)
    outliers = model.fit_predict(numeric_data)
    data['Outlier'] = (outliers == -1)
    return data

def perform_clustering(data):
    """Perform KMeans clustering on numeric data."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        console.log("[yellow]Insufficient numeric features for clustering.")
        return data

    console.log("[cyan]Performing clustering...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data

def perform_pca(data):
    """Perform Principal Component Analysis (PCA) on numeric data."""
    numeric_data = data.select_dtypes(include='number')
    if numeric_data.shape[1] < 2:
        console.log("[yellow]Insufficient numeric features for PCA.")
        return data

    console.log("[cyan]Performing PCA...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    data['PCA1'] = components[:, 0]
    data['PCA2'] = components[:, 1]

    # Scatter plot for PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='tab10')
    plt.title("PCA Scatterplot")
    plt.savefig("pca_scatterplot.png")
    plt.close()

    return data

def visualize_data(data):
    """Generate advanced visualizations."""
    numeric_data = data.select_dtypes(include='number')

    if not numeric_data.empty:
        console.log("[cyan]Generating correlation heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

        console.log("[cyan]Generating boxplot...")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=numeric_data)
        plt.title("Boxplot of Numeric Data")
        plt.savefig("boxplot.png")
        plt.close()

        console.log("[cyan]Generating histograms...")
        numeric_data.hist(figsize=(12, 10), bins=20, color='teal')
        plt.savefig("histograms.png")
        plt.close()

    else:
        console.log("[yellow]No numeric data available for visualizations.")

    return

def write_readme():
    """Write README.md to the current working directory."""
    console.log("[cyan]Generating README.md...")
    readme_content = """# Analysis Outputs

This directory contains the outputs of the analysis:

- *correlation_heatmap.png*: A heatmap showing correlations between numeric variables.
- *boxplot.png*: Boxplots of numeric variables for outlier detection.
- *histograms.png*: Histograms of numeric data distribution.
- *pca_scatterplot.png*: A scatterplot of the first two PCA components.

## Notes
- Generated using the autolysis.py script.
- Insights are enriched with LLM-powered descriptions.
"""
    with open("README.md", "w") as f:
        f.write(readme_content)
    console.log("[green]README.md generated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.log("[red]Usage: python autolysis.py <path_to_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        console.log(f"[red]File not found: {input_file}")
        sys.exit(1)

    data = load_dataset(input_file)
    data = clean_data(data)
    data = detect_outliers(data)
    data = perform_clustering(data)
    data = perform_pca(data)
    visualize_data(data)
    write_readme()

    console.log("[green]Analysis completed successfully.")

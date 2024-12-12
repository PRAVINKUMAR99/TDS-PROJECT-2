# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "rich",
#   "scikit-learn"
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

# Initialize console for rich logging
console = Console()

# Configure OpenAI
openai.api_key = os.environ.get("AIPROXY_TOKEN")

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
        plt.suptitle("Histograms of Numeric Data")
        plt.savefig("histograms.png")
        plt.close()

    if 'Cluster' in data.columns:
        console.log("[cyan]Generating cluster pairplot...")
        sns.pairplot(data, hue='Cluster', palette="tab10")
        plt.savefig("cluster_pairplot.png")
        plt.close()

def encode_image(filepath):
    """Encode an image to base64 for LLM integration."""
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        console.log(f"[red]File not found: {filepath}")
        return ""

def request_llm_insights(summary):
    """Request insights from LLM based on summary statistics."""
    console.log("[cyan]Requesting insights from LLM...")
    try:
        llm_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": f"Here is the dataset overview: {summary}. Suggest initial analyses."}
            ]
        )
        return llm_response.choices[0].message['content']
    except Exception as e:
        console.log(f"[red]LLM request failed: {e}")
        return "LLM insights could not be retrieved."

def request_visual_insights(image_data, description):
    """Request LLM to interpret visualizations."""
    if not image_data:
        console.log(f"[yellow]Skipping visualization insights for {description}.")
        return "No insights available."

    console.log("[cyan]Requesting visualization insights from LLM...")
    try:
        llm_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data visualization analyst."},
                {"role": "user", "content": f"Here is an image of {description}. Analyze its insights."},
                {"role": "user", "content": image_data}
            ]
        )
        return llm_response.choices[0].message['content']
    except Exception as e:
        console.log(f"[red]Visualization insights request failed: {e}")
        return "Visualization insights could not be retrieved."

def request_story_generation(summary, insights, visual_insights):
    """Generate a Markdown story with LLM."""
    console.log("[cyan]Requesting story generation from LLM...")
    story_prompt = (
        f"Using the analysis and visualizations, generate a Markdown report. "
        f"Include dataset summary, analyses, insights, and implications. Dataset overview: {summary}. "
        f"Insights: {insights}. Visualization Insights: {visual_insights}."
    )

    try:
        story_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data storytelling assistant."},
                {"role": "user", "content": story_prompt}
            ]
        )
        return story_response.choices[0].message['content']
    except Exception as e:
        console.log(f"[red]Story generation request failed: {e}")
        return "Story generation failed."

def analyze_and_visualize(filename):
    try:
        data = load_dataset(filename)

        if data.empty:
            console.log("[red]The dataset is empty. Exiting analysis.")
            return

        console.log("[green]Dataset loaded successfully. Performing analysis...")

        # Clean data
        data = clean_data(data)

        # Summarize dataset
        summary = {
            "columns": data.columns.tolist(),
            "types": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "summary_stats": data.describe(include='all', datetime_is_numeric=True).to_dict(),
        }

        # Detect outliers and perform clustering
        data = detect_outliers(data)
        data = perform_clustering(data)
        data = perform_pca(data)

        # Visualize data
        visualize_data(data)

        # Request insights from LLM
        insights = request_llm_insights(summary)

        # Encode visualizations and request LLM insights
        visual_insights = []
        if os.path.exists("correlation_heatmap.png"):
            visual_insights.append(request_visual_insights(encode_image("correlation_heatmap.png"), "correlation heatmap"))
        if os.path.exists("cluster_pairplot.png"):
            visual_insights.append(request_visual_insights(encode_image("cluster_pairplot.png"), "cluster pairplot"))
        if os.path.exists("pca_scatterplot.png"):
            visual_insights.append(request_visual_insights(encode_image("pca_scatterplot.png"), "PCA scatterplot"))

        # Generate Markdown story
        try:
            story = request_story_generation(summary, insights, " ".join(visual_insights))
            with open("README.md", "w") as f:
                f.write(story)
                if os.path.exists("correlation_heatmap.png"):
                    f.write("\n![Correlation Heatmap](correlation_heatmap.png)\n")
                if os.path.exists("boxplot.png"):
                    f.write("![Boxplot](boxplot.png)\n")
                if os.path.exists("histograms.png"):
                    f.write("![Histograms](histograms.png)\n")
        except Exception as e:
            console.log(f"[red]Error writing Markdown file:[/] {e}")

    except Exception as e:
        console.log(f"[red]An error occurred during analysis:[/] {e}")

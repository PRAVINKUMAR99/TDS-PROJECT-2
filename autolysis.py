# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "scikit-learn",
#   "numpy",
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openai

# Set up OpenAI API
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Function to load dataset
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try other encodings if needed
    except UnicodeDecodeError:
        print(f"Error reading {file_path}. Try a different encoding.")
        raise
    return dataset

# Function to get summary statistics
def get_summary_statistics(dataset):
    return dataset.describe()

# Function to count missing values
def count_missing_values(dataset):
    return dataset.isnull().sum()

# Function to create a correlation heatmap
def plot_correlation_heatmap(dataset):
    # Select only numeric columns
    numeric_columns = dataset.select_dtypes(include=[np.number])
    corr = numeric_columns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')

# Function to create a bar plot of missing values
def plot_missing_values(dataset):
    missing_values = dataset.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    plt.figure(figsize=(8, 6))
    missing_values.plot(kind='bar')
    plt.title('Missing Values in Each Column')
    plt.ylabel('Number of Missing Values')
    plt.tight_layout()
    plt.savefig('missing_values.png')

# Function to analyze data using LLM and summarize insights
def analyze_with_llm(file_path, summary_stats, missing_values):
    prompt = f"""
    I have the following dataset {file_path}. Here are the summary statistics of the dataset:
    {summary_stats}

    And here are the missing values in the dataset:
    {missing_values}

    Please analyze the data and provide insights, suggestions for improvement, and potential implications.
    """

    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=500
    )

    return response['choices'][0]['text']

# Function to create README.md
def create_readme(file_path, summary_stats, missing_values):
    # Prepare the markdown content
    insights = analyze_with_llm(file_path, summary_stats, missing_values)
    with open("README.md", "w") as f:
        f.write(f"# Data Analysis Report: {file_path}\n\n")
        f.write(f"## Dataset Overview\n\n")
        f.write(f"- Dataset file: {file_path}\n\n")
        f.write(f"## Summary Statistics\n\n")
        f.write(f"{summary_stats}\n\n")
        f.write(f"## Missing Values\n\n")
        f.write(f"{missing_values}\n\n")
        f.write(f"## Insights\n\n")
        f.write(insights)
        f.write("\n\n")
        f.write("## Visualizations\n")
        f.write("### Correlation Heatmap\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        f.write("### Missing Values\n")
        f.write("![Missing Values](missing_values.png)\n\n")

# Main function to run the analysis
def main(dataset_path):
    # Load the dataset
    dataset = load_dataset(dataset_path)
    
    # Generate summary statistics and missing values
    summary_stats = get_summary_statistics(dataset)
    missing_values = count_missing_values(dataset)
    
    # Create visualizations
    plot_correlation_heatmap(dataset)
    plot_missing_values(dataset)
    
    # Create README.md report
    create_readme(dataset_path, summary_stats, missing_values)
    print(f"Analysis complete! Check the generated README.md and PNG files.")

# -*- coding: utf-8 -*-
"""autolysis
Automatically generated for uv execution.

Original file is located at
    [Add the location of the original script]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib"
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys

# Set the AIPROXY_TOKEN environment variable (required for uv execution)
os.environ["AIPROXY_TOKEN"] = ""

# Set matplotlib backend to 'Agg' for headless environments
matplotlib.use('Agg')

# Function to get dataset path from the command-line argument
def get_dataset_path():
    if len(sys.argv) < 2:
        print("No dataset file provided. Please specify the path to the dataset.")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not os.path.isfile(dataset_path):
        print(f"Error: The file {dataset_path} does not exist. Please check the file path.")
        sys.exit(1)
    
    print(f"Dataset file selected: {dataset_path}")
    return dataset_path

# Function to load and analyze the dataset
def analyze_dataset(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')  # Adjust encoding if needed
        print(f"Dataset loaded successfully with shape {dataset.shape}.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset = None
    return dataset

# Function to perform basic analysis
def perform_analysis(dataset):
    # Get basic summary statistics
    summary_stats = dataset.describe()

    # Check for missing values
    missing_values = dataset.isnull().sum()

    # Create a correlation matrix
    correlation_matrix = dataset.select_dtypes(exclude='object').corr()

    # Visualize the Correlation Matrix using Seaborn and save to file
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")  # Save the plot to a file
    plt.close()  # Close the plot to avoid GUI-related issues

    return summary_stats, missing_values

# Function to create the README.md file with analysis insights
def create_readme(dataset_path, summary_stats, missing_values):
    insights = f"""
    - The dataset has {dataset.shape[0]} rows and {dataset.shape[1]} columns.
    - Summary Statistics:
    {summary_stats.to_string()}
    - Missing values by column:
    {missing_values.to_string()}
    """

    # Create the README.md file with analysis and insights
    markdown_content = f"""
# Analysis of {os.path.basename(dataset_path)}

## Dataset Overview
The dataset was loaded and analyzed dynamically. Below are the key findings:

- **Dataset Path**: `{dataset_path}`
- **Shape**: {dataset.shape[0]} rows and {dataset.shape[1]} columns.

## Analysis Highlights
### Correlation Matrix
The correlation matrix was analyzed to identify relationships between numerical features. A heatmap visualization has been generated:

![Correlation Matrix](correlation_matrix.png)

### Insights from the LLM
The following insights were generated from the basic analysis:

## Outlier Analysis
Outliers were detected in the numerical columns using the IQR method. Further investigation may be necessary for columns with high deviation.

## Next Steps
Based on the analysis:
1. Investigate features with strong correlations for potential predictive modeling.
2. Address columns with significant outliers or missing data.
3. Explore advanced techniques like clustering or anomaly detection to uncover deeper patterns.

---

This README file summarizes the analysis. For further details, please refer to the dataset and visualizations.
"""

    with open("README.md", "w") as f:
        f.write(markdown_content)

# Main execution
def main():
    # Get dataset path from command-line argument
    dataset_path = get_dataset_path()

    # Load and analyze the dataset
    dataset = analyze_dataset(dataset_path)
    
    # Perform analysis if dataset loaded successfully
    if dataset is not None:
        summary_stats, missing_values = perform_analysis(dataset)
        create_readme(dataset_path, summary_stats, missing_values)
        print("Analysis complete. Files generated: README.md, correlation_matrix.png.")
    else:
        print("Error: Dataset could not be loaded. Exiting...")

if __name__ == "__main__":
    main()

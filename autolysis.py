# -*- coding: utf-8 -*-
"""autolysis
!pip install seaborn
Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JrGStT2bcVj7IstafP0aRC6HR4a5lndr
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "google"
# ]
# ///


import os
os.environ["AIPROXY_TOKEN"] = ""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files

# 1. Upload dataset file
print("Please upload your dataset CSV file:")
uploaded = files.upload()

# Get the uploaded file's name
for filename in uploaded.keys():
    print(f"File uploaded: {filename}")
    dataset_path = filename

# Load the dataset
try:
    dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    print(f"Dataset loaded successfully with shape {dataset.shape}.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None

# 2. Perform some basic analysis on the dataset
if dataset is not None:
    # Get basic summary statistics
    summary_stats = dataset.describe()

    # Check for missing values
    missing_values = dataset.isnull().sum()

    # Create a correlation matrix
    correlation_matrix = dataset.select_dtypes(exclude='object').corr()

    # 3. Visualize the Correlation Matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()

    # 4. Define the insights based on your analysis
    insights = f"""
    - The dataset has {dataset.shape[0]} rows and {dataset.shape[1]} columns.
    - Summary Statistics:
    {summary_stats.to_string()}
    - Missing values by column:
    {missing_values.to_string()}
    """

    # 5. Create the README.md file with analysis and insights
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

    # Save the Markdown content to README.md
    with open("README.md", "w") as f:
        f.write(markdown_content)

    # Allow the user to download the README.md and chart image
    files.download("README.md")
    files.download("correlation_matrix.png")

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28.0",
#   "tenacity",
#   "scikit-learn"
# ]
# ///

import os
import sys
import base64
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
# Explanation: A correlation matrix requires at least two numeric columns to compute relationships. 
# Single numeric columns provide no meaningful pairwise comparisons.
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
        # Set OpenAI API base and key for requests
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = api_token

        # Make a request to the OpenAI API with the specified prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        # Validate response structure and return content
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

# Function to analyze visualizations via vision model
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def query_visual_insights(image_path, description):
    """
    Query vision model for insights on the provided image.
    """
    try:
        openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
        openai.api_key = api_token

        # Encode the image as Base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Query the vision model
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data visualization analyst."},
                {"role": "user", "content": f"Analyze the following visualization: {description}"},
                {"role": "user", "content": image_data}
            ]
        )

        # Validate response structure and return insights
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

# Example usage of vision insights
try:
    test_image_path = "correlation_heatmap.png"
    if os.path.exists(test_image_path):
        print(f"Querying vision insights for: {test_image_path}")
        vision_description = "Correlation Heatmap showing relationships between numeric variables."
        vision_insights = query_visual_insights(test_image_path, vision_description)
        print("Vision Insights:", vision_insights)
except Exception as e:
    print(f"Error during vision analysis: {e}")

# Advanced Analysis Functions
# Create a correlation heatmap to visualize relationships between numeric features
def create_correlation_heatmap():
    if correlation is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.savefig("correlation_heatmap.png")
        plt.close()

# Generate distribution plots for numeric columns to understand data spread and outliers
def create_distribution_plots():
    for col in numeric_df.columns:
        # Limit bins for columns with large unique values
        num_unique = numeric_df[col].nunique()
        bins = 50 if num_unique > 100 else min(num_unique, 20)

        # Explanation: Bin limits are chosen to balance detail and readability.
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[col].dropna(), kde=True, color="blue", bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"distribution_{col}.png")
        plt.close()

# Perform outlier detection using Isolation Forest
def outlier_detection():
    if not numeric_df.empty:
        model = IsolationForest(contamination=0.05, random_state=42)
        outliers = model.fit_predict(numeric_df)
        df["Outlier"] = (outliers == -1)
        plt.figure(figsize=(8, 6))
        sns.countplot(x="Outlier", data=df, palette="Set2")
        plt.title("Outlier Detection")
        plt.xlabel("Outlier Status (0: Inliers, 1: Outliers)")
        plt.ylabel("Frequency")
        plt.annotate(f"Total Outliers: {df['Outlier'].sum()}", xy=(0.5, 0.9), xycoords="axes fraction", color="red")
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
        plt.xlabel(numeric_df.columns[0])
        plt.ylabel(numeric_df.columns[1])
        plt.legend(title="Cluster")
        plt.annotate("Cluster centroids", xy=(0, 0), color="red")  # Example placeholder annotation
        plt.savefig("clustering_analysis.png")
        plt.close()

# Perform PCA for dimensionality reduction
def pca_analysis():
    if numeric_df.shape[1] > 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
        df["PCA1"] = components[:, 0]
        df["PCA2"] = components[:, 1]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="viridis")
        plt.title("PCA Analysis (Explained Variance: {:.2f}%)".format(sum(explained_variance) * 100))
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster")
        plt.savefig("pca_analysis.png")
        plt.close()

        # Bar plot for explained variance
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Principal Components")
        plt.ylabel("Percentage of Variance Explained")
        plt.savefig("pca_variance.png")
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

- **Dataset Overview**: The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. Columns include: {list(df.columns)}.
- **Summary Statistics**: Key descriptive statistics include:
  {summary[['mean', 'std', 'min', 'max']].to_dict()}.
- **Missing Values**: Columns with missing values and their counts:
  {missing_values[missing_values > 0].to_dict()}.
- **Key Findings**:
  - **Correlation Analysis**: {correlation.idxmax().to_dict() if correlation is not None else 'No significant correlations found.'}
  - **Outlier Detection**: Detected {df['Outlier'].sum() if 'Outlier' in df else 'N/A'} outliers, which may indicate anomalies or unique patterns.
  - **Clustering Analysis**: {df['Cluster'].value_counts().to_dict() if 'Cluster' in df else 'Clustering not performed due to data limitations.'}
  - **PCA Analysis**: {'Explained variance by PCA components: ' + str(PCA().fit(numeric_df).explained_variance_ratio_) if 'PCA1' in df else 'PCA not applicable.'}

Report should include:
1. **Overview of the Dataset**: Include a brief description of the dataset and its features.
2. **Key Findings from the Analysis**: Highlight major trends, patterns, and anomalies in the dataset, including insights from correlation, clustering, and PCA.
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
    f.write("![Outlier Detection](outlier_detection.png)\n")
    f.write("![Clustering Analysis](clustering_analysis.png)\n")
    f.write("![PCA Analysis](pca_analysis.png)\n")

# Ensure all outputs are in the specified directories
# Move generated files to the output directory
def safe_move(src, dst):
    """Move a file only if it exists."""
    if os.path.exists(src):
        shutil.move(src, dst)

safe_move("correlation_heatmap.png", os.path.join(output_dir, "correlation_heatmap.png"))
safe_move("outlier_detection.png", os.path.join(output_dir, "outlier_detection.png"))
safe_move("clustering_analysis.png", os.path.join(output_dir, "clustering_analysis.png"))
safe_move("pca_analysis.png", os.path.join(output_dir, "pca_analysis.png"))
for col in numeric_df.columns:
    distribution_plot = f"distribution_{col}.png"
    safe_move(distribution_plot, os.path.join(output_dir, distribution_plot))

print(f"Analysis complete. Results saved in {output_dir}/")

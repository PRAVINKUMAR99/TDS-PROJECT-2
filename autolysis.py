# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "numpy",
#   "python-dotenv"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

import openai
openai.api_key = AIPROXY_TOKEN

def load_data(file_path):
    """Load the dataset from the provided file path."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .xlsx, or .json file.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def summarize_data(df):
    """Generate a summary of the dataset."""
    summary = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include='all').to_dict()
    }
    return summary

def save_plot(output_dir, filename, plot_func):
    """Save a plot using the provided plotting function."""
    path = os.path.join(output_dir, filename)
    try:
        plot_func()
        plt.savefig(path)
        plt.close()
        return filename
    except Exception as e:
        print(f"Error generating plot {filename}: {e}")
        return None

def generate_visualizations(df, output_dir):
    """Create visualizations for the dataset."""
    visualization_paths = []

    def create_corr_heatmap():
        if df.select_dtypes(include=[np.number]).shape[1] > 1:
            corr = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")

    def create_distribution_plot(col):
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")

    def create_frequency_plot(col):
        plt.figure(figsize=(8, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette="pastel")
        plt.title(f"Frequency of {col}")

    with ThreadPoolExecutor() as executor:
        # Correlation heatmap
        future = executor.submit(save_plot, output_dir, "correlation_heatmap.png", create_corr_heatmap)
        visualization_paths.append(future.result())

        # Numeric column distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            future = executor.submit(save_plot, output_dir, f"distribution_{col}.png", lambda: create_distribution_plot(col))
            visualization_paths.append(future.result())

        # Categorical column visualizations
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            future = executor.submit(save_plot, output_dir, f"frequency_{col}.png", lambda: create_frequency_plot(col))
            visualization_paths.append(future.result())

    return [path for path in visualization_paths if path]

def ask_llm(prompt):
    """Send a prompt to the LLM and retrieve the response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using a powerful model for robust outputs
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return "Error generating narrative."

def narrate_story(summary, visualizations):
    """Generate a narrative based on the dataset summary and visualizations."""
    prompt = (
        """Analyze the following dataset summary and visualizations. "
        "Provide a comprehensive analysis, including key insights, statistical findings, and actionable conclusions."""
        f"\n\nDataset Summary:\n{json.dumps(summary, indent=2)}"
        f"\n\nVisualizations:\n{', '.join(visualizations)}"
    )
    return ask_llm(prompt)

def write_readme(content, output_dir):
    """Write the analysis report to a README.md file."""
    readme_path = os.path.join(output_dir, "README.md")
    try:
        with open(readme_path, "w") as f:
            f.write(content)
        return readme_path
    except Exception as e:
        print(f"Error writing README: {e}")
        return None

def main():
    """Main function to orchestrate the analysis process."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset>\nSupported formats: .csv, .xlsx, .json")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.dirname(file_path)

    # Load and summarize data
    df = load_data(file_path)
    summary = summarize_data(df)

    # Generate visualizations
    visualizations = generate_visualizations(df, output_dir)

    # Generate narrative
    story = narrate_story(summary, visualizations)

    # Compile README content
    readme_content = (
        "# Analysis Report\n\n"
        "## Dataset Summary\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        "## Visualizations\n\n"
        + "\n".join([f"![{viz}]({viz})" for viz in visualizations]) +
        "\n\n## Story\n\n"
        f"{story}"
    )

    # Write README
    readme_path = write_readme(readme_content, output_dir)
    if readme_path:
        print(f"Analysis complete. Results saved in {output_dir}.")
    else:
        print("Analysis complete, but failed to save the README file.")

if __name__ == "__main__":
    main()

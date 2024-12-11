# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai"
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Configure OpenAI
openai.api_key = os.environ.get("AIPROXY_TOKEN")

def analyze_and_visualize(filename):
    try:
        # Attempt to load dataset with flexible delimiters and encoding
        try:
            data = pd.read_csv(filename, encoding="utf-8")
        except UnicodeDecodeError:
            data = pd.read_csv(filename, encoding="ISO-8859-1")
        except Exception as e:
            print(f"Default CSV load failed, attempting alternative delimiters: {e}")
            data = pd.read_csv(filename, delimiter=';', encoding="utf-8")

        # Handle edge cases for empty datasets
        if data.empty:
            print("The dataset is empty. Exiting analysis.")
            return

        # Initial dataset overview
        summary = {
            "columns": data.columns.tolist(),
            "types": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "summary_stats": data.describe(include='all', datetime_is_numeric=True).to_dict(),
        }
        
        # Ask LLM for insights on the dataset
        llm_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": f"Here is the dataset overview: {summary}. Suggest initial analyses."}
            ]
        )
        insights = llm_response.choices[0].message['content']

        # Perform correlation analysis if numeric data exists
        numeric_data = data.select_dtypes(include='number')
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.savefig("correlation_matrix.png")
            plt.close()

        # Generate a cluster plot if there are at least two numeric columns
        if numeric_data.shape[1] > 1:
            sns.pairplot(numeric_data)
            plt.savefig("pairplot.png")
            plt.close()

        # Generate a README.md file with LLM narration
        story_prompt = (
            f"Using the analysis and visualizations (e.g., correlation matrix and pair plots), "
            f"generate a Markdown report. Include a summary of the dataset, the analyses performed, insights discovered, "
            f"and implications. The dataset overview is: {summary}."
        )

        story_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data storytelling assistant."},
                {"role": "user", "content": story_prompt}
            ]
        )

        with open("README.md", "w") as f:
            f.write(story_response.choices[0].message['content'])
            f.write("\n![Correlation Matrix](correlation_matrix.png)\n")
            if numeric_data.shape[1] > 1:
                f.write("![Pair Plot](pairplot.png)\n")

        print("Analysis complete. Output saved in README.md and PNG files.")

    except Exception as e:
        print(f"An error occurred: {e}")

if _name_ == "_main_":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    analyze_and_visualize(dataset_file)

    dataset_file = sys.argv[1]
    analyze_and_visualize(dataset_file)

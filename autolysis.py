import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

def analyze_numerical_data(df):
    """
    Analyze numerical columns for summary statistics and visualizations.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    print("Numerical Data Summary:")
    print(numeric_cols.describe())

    # Heatmap of correlations
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def analyze_text_data(df, text_col):
    """
    Perform basic text analysis.
    """
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found for text analysis.")
        return

    # Generate word cloud
    text_data = " ".join(df[text_col].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

def cluster_text_data(df, text_col):
    """
    Cluster text data using TF-IDF and visualize using PCA.
    """
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found for clustering.")
        return

    text_data = df[text_col].dropna()

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(text_data)

    # Apply PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap="viridis", marker="o")
    plt.title("Text Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

def main(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Analyze numerical data
    analyze_numerical_data(df)

    # Analyze text data
    text_col = "description"  # Replace with the actual column name for descriptions
    analyze_text_data(df, text_col)

    # Cluster text data
    cluster_text_data(df, text_col)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        main(file_path)
    else:
        print("Usage: python script_name.py path_to_dataset.csv")

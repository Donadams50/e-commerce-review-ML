import matplotlib.pyplot as plt
import pandas as pd

def plot_category_distribution(df):
    plt.figure(figsize=(10, 6))
    df['category'].value_counts().plot(kind='bar')
    plt.title('Distribution of Reviews across Different Categories')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

def plot_review_length_vs_categories(df):
    """
    Plots the correlation between the length of reviews content  and their assigned categories.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['content'].apply(len), df['category'])
    plt.title('Length of Reviews Contnt  vs. Assigned Categories')
    plt.xlabel('Review Length')
    plt.ylabel('Categories')
    plt.show()


if __name__ == "__main__":
    # Load the split datasets
    train_data = pd.read_csv("./split_data/train_data.csv")

    # Perform univariate EDA on the training data
    plot_category_distribution(train_data)
    plot_review_length_vs_categories(train_data)
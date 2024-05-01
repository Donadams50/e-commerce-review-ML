import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Bivariate visualization of the distribution of ratings across different categories
def visualize_rating_distribution(data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='category', y='score', data=data)
    plt.title('Distribution of Ratings across Different Categories')
    plt.xlabel('Category')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)
    plt.show()

# Bivariate visualization of the relationship between 'score' and 'thumbsUpCount'
def visualize_score_thumbsUpCount_relationship(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='score', y='thumbsUpCount', data=data)
    plt.title('Relationship between Score and Thumbs Up Count')
    plt.xlabel('Score')
    plt.ylabel('Thumbs Up Count')
    plt.show()

# Bivariate visualization of the relationship between 'score' and 'category'
def visualize_score_category_relationship(data):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='category', y='score', data=data)
    plt.title('Relationship between Score and Category')
    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Load the split datasets
    train_data = pd.read_csv("./split_data/train_data.csv")
   
visualize_rating_distribution(train_data)
visualize_score_thumbsUpCount_relationship(train_data)
visualize_score_category_relationship(train_data)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Multivariate visualization of the relationship between 'score', 'thumbsUpCount', and 'category'
def visualize_score_thumbsUpCount_category_relationship(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='score', y='thumbsUpCount', hue='category', data=data)
    plt.title('Multivariate Relationship between Score, Thumbs Up Count, and Category')
    plt.xlabel('Score')
    plt.ylabel('Thumbs Up Count')
    plt.legend(title='Category')
    plt.show()

# Multivariate visualization of the relationship between 'score', 'thumbsUpCount', and 'country'
def visualize_score_thumbsUpCount_country_relationship(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='score', y='thumbsUpCount', hue='country', data=data)
    plt.title('Multivariate Relationship between Score, Thumbs Up Count, and Country')
    plt.xlabel('Score')
    plt.ylabel('Thumbs Up Count')
    plt.legend(title='Country')
    plt.show()

# Multivariate visualization of the relationship between 'score', 'thumbsUpCount', 'category', and 'country'
def visualize_score_thumbsUpCount_category_country_relationship(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='score', y='thumbsUpCount', hue='category', style='country', data=data)
    plt.title('Multivariate Relationship between Score, Thumbs Up Count, Category, and Country')
    plt.xlabel('Score')
    plt.ylabel('Thumbs Up Count')
    plt.legend(title='Category')
    plt.show()

# Multivariate visualization of the relationship between 'score', 'thumbsUpCount', 'category', and 'validated_by'
def visualize_score_thumbsUpCount_category_validatedBy_relationship(data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='score', y='thumbsUpCount', hue='category', style='validated_by', data=data)
    plt.title('Multivariate Relationship between Score, Thumbs Up Count, Category, and Validated By')
    plt.xlabel('Score')
    plt.ylabel('Thumbs Up Count')
    plt.legend(title='Category')
    plt.show()


if __name__ == "__main__":
    # Load the split datasets
    train_data = pd.read_csv("./split_data/train_data.csv")
   
visualize_score_thumbsUpCount_category_relationship(train_data)
visualize_score_thumbsUpCount_country_relationship(train_data)
visualize_score_thumbsUpCount_category_country_relationship(train_data)
visualize_score_thumbsUpCount_category_validatedBy_relationship(train_data)

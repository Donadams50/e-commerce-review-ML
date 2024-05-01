import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

def train_model(df):
    # Training feature and target of the model
    X_train = df.drop(columns=['category'])
    y_train = df['category']

    # Create and fit the model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

 
    # Visualising the tree
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=X_train.columns, class_names=model.classes_)
    plt.show()

if __name__ == "__main__":
    # Load the split datasets
    train_data = pd.read_csv("./encoded_dataset/df_encoded.csv")
   
train_model(train_data)


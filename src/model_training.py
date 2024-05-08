import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


def train_model(df):
    # Training feature and target of the model
    X_train = df.drop(columns=['category'])
    y_train = df['category']

    # Create and fit the model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

   

    # Make predictions on the training set
    predictions = model.predict(X_train)

    # Compute the confusion matrix
    cm = confusion_matrix(y_train, predictions)
    print("Confusion Matrix:\n", cm)

  
    # Generate classification report
    report = classification_report(y_train, predictions)
    print("Classification Report:\n", report)

    # Plot the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 10}, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

 
    #Visualising the tree
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=X_train.columns, class_names=model.classes_)
    plt.show()

from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model_on_test(model, X_test, y_test):
    """
        Evaluate the trained model on the test dataset.
    """
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    # Get the classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    
if __name__ == "__main__":
    # Load the split datasets
    train_data = pd.read_csv("./encoded_dataset/df_encoded.csv")
    test_data = pd.read_csv("./encoded_dataset/test_df_encoded.csv")
   
trained_model = train_model(test_data)
test_output = evaluate_model_on_test(trained_model, test_data )



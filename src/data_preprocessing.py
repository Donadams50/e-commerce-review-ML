# data_preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import copy

def load_dataset(file_path):
    # Load the dataset from Excel file
    df = pd.read_excel(file_path)
    return df

def clean_data(df):
    # Standardize category values
    df['category'] = df['category'].str.lower().str.strip().str.replace('.', '')
    return df

def split_dataset_once(df, save_folder="split_data"):
    # Clean the dataset
    cleaned_df = clean_data(df)

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a folder for saving split datasets
    save_path = os.path.join(script_dir, save_folder)
    os.makedirs(save_path, exist_ok=True)

    # Split into intermediate and testing subsets
    intermediate_data, test_data = train_test_split(cleaned_df, test_size=0.1, random_state=42)
    # Further split intermediate subset into training and validation subsets
    train_data, val_data = train_test_split(intermediate_data, test_size=0.111, random_state=42)
    # Hide the category column in the test dataset
    test_data_hidden = test_data.drop(columns=["category"])
    # Deep copies of train, validation, and test data
    train_data_preserved = copy.deepcopy(train_data)
    val_data_preserved = copy.deepcopy(val_data)
    test_data_preserved = copy.deepcopy(test_data_hidden)
    
    # Save split datasets to separate files
    train_data_preserved.to_csv(os.path.join(save_path, "train_data.csv"), index=False)
    val_data_preserved.to_csv(os.path.join(save_path, "val_data.csv"), index=False)
    test_data_preserved.to_csv(os.path.join(save_path, "test_data_hidden.csv"), index=False)

    return train_data, val_data, test_data_hidden, train_data_preserved, val_data_preserved, test_data_preserved

if __name__ == "__main__":
    # Define path to the Excel dataset file
    file_path = "./reviews.xlsx"

    # Load the dataset
    dataset = load_dataset(file_path)

    # Split the dataset
train_data, val_data, test_data_hidden, train_data_preserved, val_data_preserved, test_data_preserved = split_dataset_once(dataset)

# Display shapes of the datasets to verify the split
print("Training data shape:", train_data.shape)
print("Deep copy Training data shape:", train_data_preserved.shape)
print("Validation data shape:", val_data.shape)
print("Deep copy Validation data shape:", val_data_preserved.shape)
print("Testing data shape (with category hidden):", test_data_hidden.shape)
print("Deep copy Testing data shape (with category hidden):", test_data_preserved.shape)

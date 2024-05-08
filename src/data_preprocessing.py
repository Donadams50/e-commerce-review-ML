import os
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(file_path):
    # Load the dataset from Excel file
    #df = pd.read_excel(file_path)
    df = pd.read_csv(file_path)
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

def print_missing_values(df):
    # Dealing with Missing Values
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

    return df


def encode_dataset(dff):
    # Drop the 'validated_by' column
    df = dff.drop(columns=["validated_by"])

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

    # Fit and transform the 'content' column
    content_features = tfidf_vectorizer.fit_transform(df['content'])

    # Convert the transformed features into a DataFrame
    content_df = pd.DataFrame(content_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Encode the 'country' column using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['country'], drop_first=False)

    # Concatenate the transformed features with the original DataFrame
    df_encoded = pd.concat([df_encoded.drop(columns=['content']), content_df], axis=1)
    
    #Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a folder for saving encoded datasets
    save_path = os.path.join(script_dir, "encoded_dataset")
    os.makedirs(save_path, exist_ok=True)
    df_encoded.to_csv(os.path.join(save_path, "test_df_encoded.csv"), index=False)

def data_preprocessing(df):
  
    # Check for missing values
    first_10_rows = df.head(10)
    
    print(first_10_rows)

    return df



if __name__ == "__main__":
    # Define path to the Excel dataset file
    #file_path = "./reviews.xlsx" 
    file_path = "./split_data/val_data.csv"

    # Load the dataset
    # dataset = load_dataset(file_path)

    file_path = "./encoded_dataset/df_encoded.csv"
    dataset = load_dataset(file_path)
    # Dealing with Missing Values
    # Split the dataset
train_data, val_data, test_data_hidden, train_data_preserved, val_data_preserved, test_data_preserved = split_dataset_once(dataset)

# Display shapes of the datasets to verify the split
print("Training data shape:", train_data.shape)
print("Deep copy Training data shape:", train_data_preserved.shape)
print("Validation data shape:", val_data.shape)
print("Deep copy Validation data shape:", val_data_preserved.shape)
print("Testing data shape (with category hidden):", test_data_hidden.shape)
print("Deep copy Testing data shape (with category hidden):", test_data_preserved.shape)


# Apply data preprocessing function
df = print_missing_values(dataset)
df_encoded = encode_dataset(dataset)
dff = data_preprocessing(dataset)

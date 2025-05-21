from model import TransformerModel
import pandas as pd

def process_data(file_path):
    """
    Function to load and process the dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print("Dataset Info:")
    print(data.info())
    
    # Display the first few rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Example: Check for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())
    
    # Example: Perform basic statistics
    print("\nBasic statistics:")
    print(data.describe())
    
    return data

if __name__ == "__main__":
    # Replace 'data.csv' with the actual path to your dataset
    file_path = "data.csv"
    processed_data = process_data(file_path)
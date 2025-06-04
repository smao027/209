from model import TransformerModel
import pandas as pd
import argparse  # Import argparse for parsing command-line arguments

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process dataset and specify model settings.")
    parser.add_argument("--file_path", type=str, default="data.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--model", type=str, default="logistic_regression", 
                        choices=["logistic_regression", "transformer", "svm", "random_forest"],
                        help="Specify the model to use. Default is logistic regression.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the data
    processed_data = process_data(args.file_path)
    
    # Print the selected model
    print(f"\nSelected model: {args.model}")
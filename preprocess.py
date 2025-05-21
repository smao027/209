import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, as_tensor=True, test_size=0.2):
    """
    Preprocess the dataset and split it into training and testing sets.
    
    Args:
        data (pd.DataFrame): Raw dataset.
        as_tensor (bool): Whether to return the output as PyTorch tensors. Default is True.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
    
    Returns:
        tuple: Train and test sets as tensors or numpy arrays.
    """
    # Handle missing values for categorical variables
    data['smoking_history'] = data['smoking_history'].fillna('No Info')

    # One-hot encode categorical variables using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
    categorical_features = data[['gender', 'smoking_history']]
    encoded_features = encoder.fit_transform(categorical_features)

    # Replace categorical columns with encoded features
    data = data.drop(columns=['gender', 'smoking_history'])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['gender', 'smoking_history']))
    data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Fill missing numerical values (if any)
    data = data.fillna(data.median())

    # Separate features and target
    X = data.drop(columns=['diabetes'])  # Features
    y = data['diabetes']  # Target

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if as_tensor:
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)
    else:
        # Return as numpy arrays
        return (X_train, y_train.values), (X_test, y_test.values)

def create_dataloaders(train_data, test_data, batch_size=32):
    """
    Create DataLoaders for training and testing.
    
    Args:
        train_data (tuple): Training data (features and labels).
        test_data (tuple): Testing data (features and labels).
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        tuple: Training and testing DataLoaders.
    """
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

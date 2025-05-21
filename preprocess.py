import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

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

def preprocess_data(data):
    """
    Preprocess the dataset.
    
    Args:
        data (pd.DataFrame): Raw dataset.
    
    Returns:
        tuple: Processed features (X) and labels (y) as tensors.
    """
    # Handle categorical data (e.g., 'gender' and 'smoking_history')
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])  # Encode 'Female' as 0, 'Male' as 1
    data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'].fillna('No Info'))  # Handle missing values

    # Fill missing numerical values (if any)
    data = data.fillna(data.median())

    # Separate features and target
    X = data.drop(columns=['diabetes'])  # Features
    y = data['diabetes']  # Target

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    return X_tensor, y_tensor

def create_dataloaders(X, y, batch_size=32, test_size=0.2):
    """
    Create DataLoaders for training and testing.
    
    Args:
        X (torch.Tensor): Features tensor.
        y (torch.Tensor): Labels tensor.
        batch_size (int): Batch size for DataLoader.
        test_size (float): Proportion of data to use for testing.
    
    Returns:
        tuple: Training and testing DataLoaders.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

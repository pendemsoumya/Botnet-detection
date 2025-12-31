"""
Data loading module for UNSW 2018 IoT Botnet dataset.
Handles CSV loading with validation and initial exploration.
"""
import pandas as pd
import numpy as np
import os

def load_dataset(dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"):
    """
    Load the UNSW 2018 IoT Botnet dataset from CSV file.
    
    Args:
        dataset_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}.\n"
            f"Please place UNSW_2018_IoT_Botnet_Full5pc_4.csv in the data/ directory.\n"
            f"Download from: https://research.unsw.edu.au/projects/bot-iot-dataset"
        )
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = pd.read_csv(dataset_path, low_memory=False)
    print(f"Dataset loaded successfully: {dataset.shape[0]} records, {dataset.shape[1]} features")
    
    return dataset


def get_dataset_info(dataset):
    """
    Display basic information about the dataset.
    
    Args:
        dataset (pd.DataFrame): The loaded dataset
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    info = {
        'total_records': dataset.shape[0],
        'total_features': dataset.shape[1],
        'missing_values': dataset.isnull().sum().sum(),
        'class_distribution': dataset['attack'].value_counts().to_dict() if 'attack' in dataset.columns else {}
    }
    
    # Display class distribution
    if 'attack' in dataset.columns:
        labels, count = np.unique(dataset['attack'].ravel(), return_counts=True)
        print(f"\nClass Distribution:")
        print(f"Normal Records: {count[0]}")
        print(f"Attack Records: {count[1]}")
        print(f"Imbalance Ratio: {count[1]/count[0]:.2f}:1 (Attack:Normal)")
    
    # Check for missing values
    null_sum = dataset.isnull().sum().sum()
    print(f"\nMissing Values: {null_sum}")
    
    return info

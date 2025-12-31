"""Data loading utilities for UNSW 2018 IoT Botnet dataset."""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def load_dataset(dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"):
    """
    Load dataset from CSV.
    
    Args:
        dataset_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If dataset not found
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading {dataset_path}")
    dataset = pd.read_csv(dataset_path, low_memory=False)
    logger.info(f"Loaded {dataset.shape[0]} records, {dataset.shape[1]} features")
    return dataset


def get_dataset_info(dataset):
    """Display dataset statistics."""
    if 'attack' in dataset.columns:
        labels, counts = np.unique(dataset['attack'].ravel(), return_counts=True)
        logger.info(f"Class distribution - Normal: {counts[0]}, Attack: {counts[1]}")
        logger.info(f"Imbalance ratio: {counts[1]/counts[0]:.2f}:1 (Attack:Normal)")
    
    missing = dataset.isnull().sum().sum()
    logger.info(f"Missing values: {missing}")

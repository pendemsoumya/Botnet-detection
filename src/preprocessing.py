"""Data preprocessing for botnet detection."""
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles preprocessing: encoding, normalization, SMOTE balancing."""
    
    def __init__(self):
        """Initialize encoders and scaler."""
        self.encoders = [LabelEncoder() for _ in range(7)]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def drop_irrelevant_columns(self, dataset):
        """Remove non-predictive columns."""
        return dataset.drop(['pkSeqID', 'category', 'subcategory'], axis=1)
    
    def encode_categorical_features(self, dataset):
        """Encode categorical features to numeric."""
        cat_features = ['flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'state']
        for encoder, feature in zip(self.encoders, cat_features):
            dataset[feature] = encoder.fit_transform(dataset[feature].astype(str))
        return dataset
    
    def normalize_features(self, X):
        """Normalize features to [0, 1] range."""
        return self.scaler.fit_transform(X)
    
    def apply_smote(self, X, Y):
        """
        Apply SMOTE to balance classes.
        Uses 10% sampling ratio to manage memory.
        """
        labels, counts = np.unique(Y, return_counts=True)
        target_count = int(max(counts) * 0.1)
        
        logger.info(f"SMOTE: {counts[0]} → {target_count} normal samples")
        smote = SMOTE(sampling_strategy={0: target_count}, random_state=42)
        X_balanced, Y_balanced = smote.fit_resample(X, Y)
        
        return X_balanced, Y_balanced
    
    def prepare_data(self, dataset, test_size=0.2, random_seed=None):
        """
        Full preprocessing pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        dataset = self.drop_irrelevant_columns(dataset)
        dataset = self.encode_categorical_features(dataset)
        
        Y = dataset['attack'].ravel()
        X = dataset.drop('attack', axis=1).values
        
        # Shuffle
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]
        
        # Normalize and balance
        X = self.normalize_features(X)
        X, Y = self.apply_smote(X, Y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
        logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

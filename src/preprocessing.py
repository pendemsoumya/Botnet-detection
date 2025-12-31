"""
Data preprocessing module for botnet detection.
Handles feature engineering, encoding, normalization, and SMOTE balancing.
Preserves the exact preprocessing pipeline from the original implementation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    """
    Handles all preprocessing steps for the botnet dataset.
    Maintains exact logic from original implementation for reproducibility.
    """
    
    def __init__(self):
        """Initialize preprocessor with label encoders and scaler."""
        # Create 7 separate encoders as in original implementation
        self.encoder1 = LabelEncoder()  # for 'flgs'
        self.encoder2 = LabelEncoder()  # for 'proto'
        self.encoder3 = LabelEncoder()  # for 'saddr'
        self.encoder4 = LabelEncoder()  # for 'sport'
        self.encoder5 = LabelEncoder()  # for 'daddr'
        self.encoder6 = LabelEncoder()  # for 'dport'
        self.encoder7 = LabelEncoder()  # for 'state'
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def drop_irrelevant_columns(self, dataset):
        """
        Drop columns that are not needed for classification.
        
        Args:
            dataset (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with irrelevant columns removed
        """
        # Drop pkSeqID (sequence identifier), category, and subcategory
        columns_to_drop = ['pkSeqID', 'category', 'subcategory']
        dataset_cleaned = dataset.drop(columns_to_drop, axis=1, inplace=False)
        print(f"Dropped columns: {columns_to_drop}")
        print(f"Remaining features: {dataset_cleaned.shape[1] - 1} (plus 'attack' target)")
        return dataset_cleaned
    
    def encode_categorical_features(self, dataset):
        """
        Convert categorical string features to numeric using Label Encoding.
        Uses 7 separate encoders as in the original implementation.
        
        Args:
            dataset (pd.DataFrame): Input dataset with categorical features
            
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        print("\nEncoding categorical features...")
        
        # Encode each categorical feature with its dedicated encoder
        dataset['flgs'] = pd.Series(self.encoder1.fit_transform(dataset['flgs'].astype(str)))
        dataset['proto'] = pd.Series(self.encoder2.fit_transform(dataset['proto'].astype(str)))
        dataset['saddr'] = pd.Series(self.encoder3.fit_transform(dataset['saddr'].astype(str)))
        dataset['sport'] = pd.Series(self.encoder4.fit_transform(dataset['sport'].astype(str)))
        dataset['daddr'] = pd.Series(self.encoder5.fit_transform(dataset['daddr'].astype(str)))
        dataset['dport'] = pd.Series(self.encoder6.fit_transform(dataset['dport'].astype(str)))
        dataset['state'] = pd.Series(self.encoder7.fit_transform(dataset['state'].astype(str)))
        
        print("Encoded features: flgs, proto, saddr, sport, daddr, dport, state")
        return dataset
    
    def normalize_features(self, X):
        """
        Normalize features to [0, 1] range using MinMaxScaler.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Normalized feature matrix
        """
        print("\nNormalizing features using MinMaxScaler...")
        X_normalized = self.scaler.fit_transform(X)
        print(f"Features normalized to range [0, 1]")
        return X_normalized
    
    def apply_smote(self, X, Y):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
        
        SMOTE creates synthetic samples of the minority class by:
        1. Finding k-nearest neighbors for each minority sample
        2. Randomly selecting one neighbor
        3. Creating synthetic sample along the line connecting them
        
        This addresses the severe class imbalance (0.07% normal vs 99.93% attack).
        
        Args:
            X (np.ndarray): Feature matrix
            Y (np.ndarray): Target labels
            
        Returns:
            tuple: (X_resampled, Y_resampled) balanced dataset
        """
        print("\nApplying SMOTE for class balancing...")
        labels_before, count_before = np.unique(Y, return_counts=True)
        print(f"Before SMOTE - Normal: {count_before[0]}, Attack: {count_before[1]}")
        
        smote = SMOTE()
        X_balanced, Y_balanced = smote.fit_resample(X, Y)
        
        labels_after, count_after = np.unique(Y_balanced, return_counts=True)
        print(f"After SMOTE - Normal: {count_after[0]}, Attack: {count_after[1]}")
        print(f"Dataset size increased from {len(Y)} to {len(Y_balanced)} samples")
        
        return X_balanced, Y_balanced
    
    def prepare_data(self, dataset, test_size=0.2, random_seed=None):
        """
        Complete preprocessing pipeline: encode, shuffle, normalize, balance, split.
        
        Args:
            dataset (pd.DataFrame): Raw dataset
            test_size (float): Proportion of test set (default 0.2 = 20%)
            random_seed (int): Random seed for reproducibility (default None)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) ready for model training
        """
        # Step 1: Drop irrelevant columns
        dataset = self.drop_irrelevant_columns(dataset)
        
        # Step 2: Encode categorical features
        dataset = self.encode_categorical_features(dataset)
        
        # Step 3: Separate features and target
        Y = dataset['attack'].ravel()
        data = dataset.values
        X = data[:, 0:dataset.shape[1]-1]
        
        # Step 4: Shuffle dataset (preserves original behavior)
        print(f"\nShuffling dataset...")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        
        # Step 5: Normalize features
        X = self.normalize_features(X)
        
        # Step 6: Apply SMOTE for balancing
        X, Y = self.apply_smote(X, Y)
        
        # Step 7: Split into train and test sets
        print(f"\nSplitting dataset: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        
        print(f"Total records: {X.shape[0]}")
        print(f"Total features: {X.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test

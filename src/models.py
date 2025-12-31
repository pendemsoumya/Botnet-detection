"""ML/DL models for botnet detection."""
import numpy as np
import os
import pickle
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


class DefaultDecisionTree:
    """Decision Tree classifier with max_depth=1."""
    
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=1)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)


class BOGPDecisionTree:
    """Decision Tree with Bayesian Optimization for hyperparameter tuning."""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def _objective(self, X_train, y_train, max_depth, min_samples_split, max_features):
        """Objective function for Bayesian Optimization."""
        params = {
            'max_depth': int(max_depth),
            'min_samples_split': min_samples_split,
            'max_features': max_features
        }
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=123, **params),
            X_train, y_train, cv=5
        )
        return scores.mean()
    
    def optimize_hyperparameters(self, X_train, y_train, init_points=5, n_iter=2):
        """Find optimal hyperparameters using Bayesian Optimization."""
        logger.info("Optimizing hyperparameters with BOGP")
        
        optimizer = BayesianOptimization(
            f=lambda max_depth, min_samples_split, max_features: 
                self._objective(X_train, y_train, max_depth, min_samples_split, max_features),
            pbounds={
                'max_depth': (5, 10),
                'min_samples_split': (0.1, 0.9),
                'max_features': (0.1, 0.9)
            },
            random_state=111
        )
        
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        self.best_params = optimizer.max['params']
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def train(self, X_train, y_train):
        """Train with optimized hyperparameters."""
        if self.best_params is None:
            raise ValueError("Call optimize_hyperparameters() first")
            
        self.model = DecisionTreeClassifier(
            max_depth=self.best_params['max_depth'],
            max_features=self.best_params['max_features'],
            min_samples_split=self.best_params['min_samples_split']
        )
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X_test)


class SVMClassifier:
    """Support Vector Machine with training sample limit."""
    
    def __init__(self, training_limit=50):
        self.model = svm.SVC()
        self.training_limit = training_limit
        
    def train(self, X_train, y_train):
        """Train on limited samples due to computational cost."""
        logger.info(f"Training SVM with {self.training_limit} samples")
        self.model.fit(X_train[:self.training_limit], y_train[:self.training_limit])
        
    def predict(self, X_test):
        return self.model.predict(X_test)


class CNNClassifier:
    """Convolutional Neural Network for botnet detection."""
    
    def __init__(self, model_path="models/cnn_weights.hdf5", history_path="models/cnn_history.pckl"):
        self.model = None
        self.model_path = model_path
        self.history_path = history_path
        
    def _build_model(self, input_shape, num_classes):
        """Build CNN architecture."""
        model = Sequential([
            Conv2D(32, (1, 1), input_shape=input_shape, activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),
            Conv2D(16, (1, 1), activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape_for_cnn(self, X):
        """Reshape to (samples, features, 1, 1)."""
        return np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    
    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train CNN model."""
        X_train_cnn = self._reshape_for_cnn(X_train)
        X_test_cnn = self._reshape_for_cnn(X_test)
        y_train_cnn = to_categorical(y_train)
        y_test_cnn = to_categorical(y_test)
        
        self.model = self._build_model(
            input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]),
            num_classes=y_train_cnn.shape[1]
        )
        
        if os.path.exists(self.model_path):
            logger.info(f"Loading weights from {self.model_path}")
            self.model.load_weights(self.model_path)
        else:
            logger.info(f"Training CNN for {epochs} epochs")
            checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                verbose=0,
                save_best_only=True
            )
            
            history = self.model.fit(
                X_train_cnn, y_train_cnn,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test_cnn, y_test_cnn),
                callbacks=[checkpoint],
                verbose=0
            )
            
            with open(self.history_path, 'wb') as f:
                pickle.dump(history.history, f)
    
    def predict(self, X_test):
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_test_cnn = self._reshape_for_cnn(X_test)
        predictions = self.model.predict(X_test_cnn, verbose=0)
        return np.argmax(predictions, axis=1)


class BOGPDecisionTree:
    """
    Bayesian Optimization with Gaussian Process for Decision Tree hyperparameter tuning.
    
    BOGP optimizes hyperparameters by:
    1. Building probabilistic model of objective function
    2. Using acquisition function to select promising parameters
    3. Iteratively updating belief based on observations
    
    This is more efficient than grid search for expensive evaluations.
    """
    
    def __init__(self):
        """Initialize BOGP-optimized Decision Tree."""
        self.model = None
        self.best_params = None
        
    def _gaussian_process_objective(self, X_train, y_train, max_depth, min_samples_split, max_features):
        """
        Objective function for Bayesian Optimization.
        Returns cross-validation score for given hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            max_features: Maximum features to consider
            
        Returns:
            float: Mean cross-validation score
        """
        params_dt = {
            'max_depth': int(max_depth),
            'min_samples_split': min_samples_split,
            'max_features': max_features
        }
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=123, **params_dt),
            X_train, y_train, cv=5
        )
        return scores.mean()
    
    def optimize_hyperparameters(self, X_train, y_train, init_points=5, n_iter=2):
        """
        Use Bayesian Optimization to find best hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            init_points: Number of random exploration steps
            n_iter: Number of optimization steps
            
        Returns:
            dict: Best hyperparameters found
        """
        print("\nOptimizing Decision Tree hyperparameters using BOGP...")
        print("This uses Bayesian Optimization with Gaussian Process surrogate model")
        
        # Define parameter bounds
        params_bounds = {
            'max_depth': (5, 10),
            'min_samples_split': (0.1, 0.9),
            'max_features': (0.1, 0.9)
        }
        
        # Create optimizer
        optimizer = BayesianOptimization(
            f=lambda max_depth, min_samples_split, max_features: 
                self._gaussian_process_objective(X_train, y_train, max_depth, min_samples_split, max_features),
            pbounds=params_bounds,
            random_state=111
        )
        
        # Run optimization
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        self.best_params = optimizer.max['params']
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        
        print(f"Best hyperparameters found: {self.best_params}")
        return self.best_params
    
    def train(self, X_train, y_train):
        """
        Train Decision Tree with optimized hyperparameters.
        Must call optimize_hyperparameters() first.
        """
        if self.best_params is None:
            raise ValueError("Must run optimize_hyperparameters() before training")
            
        print("\nTraining BOGP-Optimized Decision Tree...")
        self.model = DecisionTreeClassifier(
            max_depth=self.best_params['max_depth'],
            max_features=self.best_params['max_features'],
            min_samples_split=self.best_params['min_samples_split']
        )
        self.model.fit(X_train, y_train)
        print("BOGP-Optimized Decision Tree training completed")
        
    def predict(self, X_test):
        """Predict on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X_test)


class SVMClassifier:
    """
    Support Vector Machine for binary classification.
    Limited to 50 training samples due to computational constraints (from original code).
    
    SVM finds optimal hyperplane that maximizes margin between classes.
    """
    
    def __init__(self, training_limit=50):
        """
        Initialize SVM classifier.
        
        Args:
            training_limit: Maximum number of training samples to use (default 50)
        """
        self.model = svm.SVC()
        self.training_limit = training_limit
        
    def train(self, X_train, y_train):
        """
        Train SVM with limited samples.
        
        Note: Original implementation uses only 50 samples due to high computational cost
        of SVM on large datasets. This is a known limitation.
        """
        print(f"\nTraining SVM with {self.training_limit} samples...")
        print("(Limited training size due to SVM computational complexity)")
        self.model.fit(X_train[0:self.training_limit], y_train[0:self.training_limit])
        print("SVM training completed")
        
    def predict(self, X_test):
        """Predict on test data."""
        return self.model.predict(X_test)


class CNNClassifier:
    """
    Convolutional Neural Network for botnet detection.
    
    Architecture (preserves original design):
    - Input: (n_features, 1, 1) - Each network flow as 1x1 image with multiple channels
    - Conv2D: 32 filters (1x1), ReLU activation
    - MaxPooling2D: (1x1) pool size
    - Conv2D: 16 filters (1x1), ReLU activation
    - MaxPooling2D: (1x1) pool size
    - Flatten
    - Dense: 256 units, ReLU
    - Output: 2 units (Normal/Attack), Softmax
    
    Note: The (1,1) kernel size is unusual but preserved for compatibility with report.
    It effectively makes the CNN act on individual features without spatial convolution.
    """
    
    def __init__(self, model_path="models/cnn_weights.hdf5", history_path="models/cnn_history.pckl"):
        """
        Initialize CNN classifier.
        
        Args:
            model_path: Path to save/load model weights
            history_path: Path to save training history
        """
        self.model = None
        self.model_path = model_path
        self.history_path = history_path
        
    def _build_model(self, input_shape, num_classes):
        """
        Build CNN architecture.
        
        Args:
            input_shape: Shape of input (n_features, 1, 1)
            num_classes: Number of output classes (2 for binary)
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(
            32, (1, 1),
            input_shape=input_shape,
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        
        # Second convolutional block
        model.add(Conv2D(16, (1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape_for_cnn(self, X):
        """
        Reshape feature matrix to CNN-compatible 4D format.
        
        Args:
            X: Feature matrix (samples, features)
            
        Returns:
            np.ndarray: Reshaped to (samples, features, 1, 1)
        """
        return np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    
    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """
        Train CNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for validation)
            y_test: Test labels (for validation)
            epochs: Number of training epochs (default 5)
            batch_size: Batch size (default 32)
        """
        print("\nPreparing data for CNN...")
        
        # Reshape to 4D for CNN
        X_train_cnn = self._reshape_for_cnn(X_train)
        X_test_cnn = self._reshape_for_cnn(X_test)
        
        # Convert labels to categorical (one-hot encoding)
        y_train_cnn = to_categorical(y_train)
        y_test_cnn = to_categorical(y_test)
        
        print(f"CNN input shape: {X_train_cnn.shape}")
        
        # Build model
        self.model = self._build_model(
            input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]),
            num_classes=y_train_cnn.shape[1]
        )
        
        print("\nCNN Architecture:")
        print("- Conv2D Layer 1: 32 filters, (1x1) kernel, ReLU")
        print("- MaxPooling2D: (1x1) pool")
        print("- Conv2D Layer 2: 16 filters, (1x1) kernel, ReLU")
        print("- MaxPooling2D: (1x1) pool")
        print("- Flatten Layer")
        print("- Dense Layer: 256 units, ReLU")
        print("- Output Layer: 2 units, Softmax")
        
        # Check if pre-trained weights exist
        if os.path.exists(self.model_path):
            print(f"\nLoading pre-trained weights from {self.model_path}")
            self.model.load_weights(self.model_path)
        else:
            print(f"\nTraining CNN for {epochs} epochs...")
            
            # Create checkpoint callback to save best model
            checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                verbose=1,
                save_best_only=True
            )
            
            # Train model
            history = self.model.fit(
                X_train_cnn, y_train_cnn,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test_cnn, y_test_cnn),
                callbacks=[checkpoint],
                verbose=1
            )
            
            # Save training history
            with open(self.history_path, 'wb') as f:
                pickle.dump(history.history, f)
            
            print(f"Model saved to {self.model_path}")
            print(f"Training history saved to {self.history_path}")
    
    def predict(self, X_test):
        """
        Predict on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Reshape to 4D for CNN
        X_test_cnn = self._reshape_for_cnn(X_test)
        
        # Get predictions
        predictions = self.model.predict(X_test_cnn)
        
        # Convert from one-hot to class labels
        return np.argmax(predictions, axis=1)

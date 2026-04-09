"""ML and DL models for botnet detection."""
import logging
import os
import pickle

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

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
        params = {
            "max_depth": int(max_depth),
            "min_samples_split": min_samples_split,
            "max_features": max_features,
        }
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=123, **params),
            X_train,
            y_train,
            cv=5,
        )
        return scores.mean()

    def optimize_hyperparameters(self, X_train, y_train, init_points=5, n_iter=2):
        """Find optimal hyperparameters using Bayesian Optimization."""
        logger.info("Optimizing hyperparameters with BOGP")

        optimizer = BayesianOptimization(
            f=lambda max_depth, min_samples_split, max_features: self._objective(
                X_train, y_train, max_depth, min_samples_split, max_features
            ),
            pbounds={
                "max_depth": (5, 10),
                "min_samples_split": (0.1, 0.9),
                "max_features": (0.1, 0.9),
            },
            random_state=111,
        )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        self.best_params = optimizer.max["params"]
        self.best_params["max_depth"] = int(self.best_params["max_depth"])
        logger.info("Best params: %s", self.best_params)
        return self.best_params

    def train(self, X_train, y_train):
        if self.best_params is None:
            raise ValueError("Call optimize_hyperparameters() first")

        self.model = DecisionTreeClassifier(
            max_depth=self.best_params["max_depth"],
            max_features=self.best_params["max_features"],
            min_samples_split=self.best_params["min_samples_split"],
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
        logger.info("Training SVM with %s samples", self.training_limit)
        self.model.fit(X_train[: self.training_limit], y_train[: self.training_limit])

    def predict(self, X_test):
        return self.model.predict(X_test)


class CNNClassifier:
    """Convolutional Neural Network for botnet detection."""

    def __init__(self, model_path="models/cnn_weights.weights.h5", history_path="models/cnn_history.pckl"):
        self.model = None
        self.model_path = model_path
        self.history_path = history_path

    def _build_model(self, input_shape, num_classes):
        model = Sequential(
            [
                Conv2D(32, (1, 1), input_shape=input_shape, activation="relu"),
                MaxPooling2D(pool_size=(1, 1)),
                Conv2D(16, (1, 1), activation="relu"),
                MaxPooling2D(pool_size=(1, 1)),
                Flatten(),
                Dense(256, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _reshape_for_cnn(self, X):
        return np.reshape(X, (X.shape[0], X.shape[1], 1, 1))

    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train CNN model or load existing weights."""
        X_train_cnn = self._reshape_for_cnn(X_train)
        X_test_cnn = self._reshape_for_cnn(X_test)
        y_train_cnn = to_categorical(y_train)
        y_test_cnn = to_categorical(y_test)

        self.model = self._build_model(
            input_shape=(
                X_train_cnn.shape[1],
                X_train_cnn.shape[2],
                X_train_cnn.shape[3],
            ),
            num_classes=y_train_cnn.shape[1],
        )

        if os.path.exists(self.model_path):
            logger.info("Loading weights from %s", self.model_path)
            self.model.load_weights(self.model_path)
            return

        logger.info("Training CNN for %s epochs", epochs)
        checkpoint = ModelCheckpoint(
            filepath=self.model_path,
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
        )

        history = self.model.fit(
            X_train_cnn,
            y_train_cnn,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_cnn, y_test_cnn),
            callbacks=[checkpoint],
            verbose=0,
        )

        with open(self.history_path, "wb") as file_handle:
            pickle.dump(history.history, file_handle)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained")

        X_test_cnn = self._reshape_for_cnn(X_test)
        predictions = self.model.predict(X_test_cnn, verbose=0)
        return np.argmax(predictions, axis=1)

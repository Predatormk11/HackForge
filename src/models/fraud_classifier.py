"""Supervised Random Forest fraud classification model."""
import os
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_engineering import FEATURE_COLUMNS


class FraudClassifier:
    """Random Forest classifier for estimating transaction fraud probability."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42, model_path: Optional[str] = None):
        """Initialize the fraud classifier.

        Args:
            n_estimators: Number of trees in the forest.
            random_state: Seed for reproducibility.
            model_path: Optional path to load a pre-trained model.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_path = model_path or "models/fraud_model.pkl"
        self.model: Optional[RandomForestClassifier] = None
        self.is_fitted: bool = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "FraudClassifier":
        """Train the RandomForestClassifier on feature matrix X and target y.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary labels (0 for legitimate, 1 for fraudulent).

        Returns:
            self
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_probability(self, features: Union[pd.DataFrame, np.ndarray]) -> float:
        """Predict probability of fraud for a single transaction or batch.

        Args:
            features: Feature DataFrame or array.

        Returns:
            float fraud probability clamped in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None:
            # Fallback heuristic if model is not yet trained/loaded
            return 0.10

        if isinstance(features, pd.DataFrame):
            # Ensure correct column order
            cols = [c for c in FEATURE_COLUMNS if c in features.columns]
            if len(cols) == len(FEATURE_COLUMNS):
                features = features[FEATURE_COLUMNS]

        proba = self.model.predict_proba(features)
        # Check if binary classes (0, 1) exist
        if proba.shape[1] == 2:
            prob_fraud = float(proba[0, 1])
        else:
            # Single class edge case
            prob_fraud = float(proba[0, 0])

        return float(np.clip(prob_fraud, 0.0, 1.0))

    def save(self, path: Optional[str] = None) -> str:
        """Serialize and save the trained model using joblib.

        Args:
            path: Destination file path.

        Returns:
            Saved file path.
        """
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(self.model, save_path)
        return save_path

    def load(self, path: Optional[str] = None) -> "FraudClassifier":
        """Load a saved model from disk.

        Args:
            path: Path to serialized model file.

        Returns:
            self
        """
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")
        self.model = joblib.load(load_path)
        self.is_fitted = True
        return self

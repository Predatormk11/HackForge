"""Unsupervised Isolation Forest anomaly detection model."""
import os
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.feature_engineering import FEATURE_COLUMNS


class AnomalyDetector:
    """Isolation Forest model for unsupervised anomaly score estimation."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
        model_path: Optional[str] = None,
    ):
        """Initialize Anomaly Detector.

        Args:
            n_estimators: Number of isolation trees.
            contamination: Expected proportion of outliers in the training data.
            random_state: Random seed.
            model_path: Optional path to serialized model file.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model_path = model_path or "models/anomaly_model.pkl"
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, X: Union[pd.DataFrame, np.ndarray]) -> "AnomalyDetector":
        """Train IsolationForest on feature matrix X.

        Args:
            X: Feature matrix.

        Returns:
            self
        """
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, features: Union[pd.DataFrame, np.ndarray]) -> float:
        """Predict normalized anomaly score in [0.0, 1.0], where 1.0 indicates high anomaly.

        Args:
            features: Feature DataFrame or array for a transaction.

        Returns:
            Normalized anomaly score in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None:
            return 0.10

        if isinstance(features, pd.DataFrame):
            cols = [c for c in FEATURE_COLUMNS if c in features.columns]
            if len(cols) == len(FEATURE_COLUMNS):
                features = features[FEATURE_COLUMNS]

        # decision_function yields lower (negative) values for anomalies, positive for normal inliers
        raw_score = float(self.model.decision_function(features)[0])

        # Calibrate & normalize: raw ~ 0.2 (normal) -> 0.0, raw ~ -0.2 (anomaly) -> 1.0
        normalized_score = (0.20 - raw_score) / 0.40
        return float(np.clip(normalized_score, 0.0, 1.0))

    def save(self, path: Optional[str] = None) -> str:
        """Serialize and save model to disk.

        Args:
            path: Destination file path.

        Returns:
            Saved file path.
        """
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(self.model, save_path)
        return save_path

    def load(self, path: Optional[str] = None) -> "AnomalyDetector":
        """Load serialized model from disk.

        Args:
            path: Path to model file.

        Returns:
            self
        """
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Anomaly model not found at {load_path}")
        self.model = joblib.load(load_path)
        self.is_fitted = True
        return self

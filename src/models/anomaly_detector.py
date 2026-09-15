"""Unsupervised Isolation Forest anomaly detection model with statistical calibration."""
import os
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.features.feature_engineering import FEATURE_COLUMNS


class AnomalyDetector:
    """Isolation Forest model for unsupervised anomaly score estimation."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: Union[float, str] = "auto",
        random_state: int = 42,
        model_path: Optional[str] = None,
    ):
        """Initialize Anomaly Detector.

        Args:
            n_estimators: Number of isolation trees.
            contamination: Expected proportion of outliers in the training data, or "auto".
            random_state: Random seed.
            model_path: Optional path to serialized model file.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model_path = model_path or "models/anomaly_model.pkl"
        self.model: Optional[IsolationForest] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.score_scaler: Optional[MinMaxScaler] = None
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
        # 1. Scale Features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        # 2. Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)
        
        # 3. Fit Score Calibration
        # decision_function yields lower (negative) values for anomalies, positive for normal inliers
        raw_scores = self.model.decision_function(X_scaled)
        # Invert scores so higher means more anomalous
        inverted_scores = -raw_scores.reshape(-1, 1)
        
        self.score_scaler = MinMaxScaler(feature_range=(0, 1))
        self.score_scaler.fit(inverted_scores)

        self.is_fitted = True
        return self

    def predict_anomaly_score(self, features: Union[pd.DataFrame, np.ndarray]) -> float:
        """Predict normalized anomaly score in [0.0, 1.0], where 1.0 indicates high anomaly.

        Args:
            features: Feature DataFrame or array for a transaction.

        Returns:
            Normalized anomaly score in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None or self.feature_scaler is None or self.score_scaler is None:
            return 0.10

        if isinstance(features, pd.DataFrame):
            cols = [c for c in FEATURE_COLUMNS if c in features.columns]
            if len(cols) == len(FEATURE_COLUMNS):
                features = features[FEATURE_COLUMNS]

        X_scaled = self.feature_scaler.transform(features)
        
        raw_score = self.model.decision_function(X_scaled)
        inverted_score = -raw_score.reshape(-1, 1)
        
        normalized_score = self.score_scaler.transform(inverted_score)[0, 0]
        return float(np.clip(normalized_score, 0.0, 1.0))

    def save(self, path: Optional[str] = None) -> str:
        """Serialize and save model (and scalers) to disk.

        Args:
            path: Destination file path.

        Returns:
            Saved file path.
        """
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # joblib naturally serializes the entire class instance including the scalers
        joblib.dump(self, save_path)
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
        
        loaded_instance = joblib.load(load_path)
        
        self.model = loaded_instance.model
        self.feature_scaler = loaded_instance.feature_scaler
        self.score_scaler = loaded_instance.score_scaler
        self.contamination = loaded_instance.contamination
        self.n_estimators = loaded_instance.n_estimators
        
        self.is_fitted = True
        return self

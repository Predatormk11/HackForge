"""Train the unsupervised Isolation Forest anomaly detection model."""
import os
import pandas as pd

from src.features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.models.anomaly_detector import AnomalyDetector
from scripts.generate_demo_data import generate_synthetic_training_data


def train_anomaly_model(
    data_csv: str = "data/raw/train_transactions.csv",
    model_output: str = "models/anomaly_model.pkl",
) -> AnomalyDetector:
    """Train and persist the Isolation Forest anomaly detector.

    Args:
        data_csv: Path to input training data CSV.
        model_output: Output destination path for the serialized model.

    Returns:
        Trained AnomalyDetector instance.
    """
    if not os.path.exists(data_csv):
        print(f"Dataset not found at {data_csv}. Generating synthetic baseline data...")
        generate_synthetic_training_data(output_csv=data_csv)

    print(f"Loading training data from {data_csv}...")
    df = pd.read_csv(data_csv)

    # Train IsolationForest on all historical records (or predominantly legitimate records)
    feature_engineer = FeatureEngineer()
    print("Extracting feature matrix for Isolation Forest...")
    X = feature_engineer.transform_dataframe(df)

    detector = AnomalyDetector(model_path=model_output, contamination=0.08)
    detector.train(X)

    saved_path = detector.save(model_output)
    print(f"Anomaly detector saved to: {saved_path}")

    return detector


if __name__ == "__main__":
    train_anomaly_model()

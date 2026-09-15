"""Train the supervised Random Forest fraud classifier model."""
import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.models.fraud_classifier import FraudClassifier
from scripts.generate_demo_data import generate_synthetic_training_data


def train_fraud_model(
    data_csv: str = "data/raw/train_transactions.csv",
    model_output: str = "models/fraud_model.pkl",
) -> FraudClassifier:
    """Train and persist the Random Forest fraud classifier.

    Args:
        data_csv: Path to input training data CSV.
        model_output: Output destination path for the serialized model.

    Returns:
        Trained FraudClassifier instance.
    """
    if not os.path.exists(data_csv):
        print(f"Dataset not found at {data_csv}. Generating synthetic baseline data...")
        generate_synthetic_training_data(output_csv=data_csv)

    print(f"Loading training data from {data_csv}...")
    df = pd.read_csv(data_csv)

    feature_engineer = FeatureEngineer()
    print("Extracting feature matrix...")
    X = feature_engineer.transform_dataframe(df)
    y = df["is_fraud"].values

    print(f"Training set: {len(X)} samples with {len(FEATURE_COLUMNS)} features. Fraud rate: {y.mean():.1%}")

    classifier = FraudClassifier(model_path=model_output)
    classifier.train(X, y)

    # In-sample baseline diagnostic (prototype verification)
    preds = classifier.model.predict(X)
    probas = classifier.model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probas)
    print(f"Model fitted successfully. Training ROC-AUC: {auc:.4f}")

    saved_path = classifier.save(model_output)
    print(f"Fraud model saved to: {saved_path}")

    return classifier


if __name__ == "__main__":
    train_fraud_model()

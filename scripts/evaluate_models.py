import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.models.fraud_classifier import FraudClassifier
from src.models.anomaly_detector import AnomalyDetector
from src.features.feature_engineering import FeatureEngineer

def evaluate():
    data_csv = "data/raw/train_transactions.csv"
    if not os.path.exists(data_csv):
        print(f"Dataset not found at {data_csv}")
        return

    print("Loading data...")
    df = pd.read_csv(data_csv)
    
    engineer = FeatureEngineer()
    X = engineer.transform_dataframe(df)
    y = df["is_fraud"].values

    # We split to evaluate on a holdout set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n--- Evaluation Set ---")
    print(f"Test size: {len(X_test)} samples")
    print(f"Fraud rate: {y_test.mean():.1%}")

    # 1. Evaluate Fraud Classifier
    print("\n--- Fraud Classifier (Supervised) ---")
    fraud_model = FraudClassifier().load("models/fraud_model.pkl")
    
    # Predict probabilities for the test set
    # Since predict_probability only returns a single float for the first row,
    # we'll use the underlying model's predict_proba for batch evaluation.
    probas = fraud_model.model.predict_proba(X_test)[:, 1]
    preds = (probas > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, probas)
    print(f"ROC-AUC: {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # 2. Evaluate Anomaly Detector
    print("\n--- Anomaly Detector (Unsupervised) ---")
    anomaly_model = AnomalyDetector().load("models/anomaly_model.pkl")
    
    # Batch predict
    # Predict using the internal calibrated pipeline
    X_scaled = anomaly_model.feature_scaler.transform(X_test)
    raw_scores = anomaly_model.model.decision_function(X_scaled)
    inverted = -raw_scores.reshape(-1, 1)
    normalized = anomaly_model.score_scaler.transform(inverted).flatten()
    
    # We expect anomalies to have higher scores. 
    # Let's see the average score for legit vs fraud transactions
    avg_legit_score = np.mean(normalized[y_test == 0])
    avg_fraud_score = np.mean(normalized[y_test == 1])
    
    print(f"Average Anomaly Score for Legitimate: {avg_legit_score:.4f}")
    print(f"Average Anomaly Score for Fraudulent: {avg_fraud_score:.4f}")
    
    if avg_fraud_score > avg_legit_score:
        print("-> Success! Fraudulent transactions are generally scoring higher as anomalies.")
    else:
        print("-> Warning! Anomaly scores do not align with known fraud.")


if __name__ == "__main__":
    evaluate()

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.models.fraud_classifier import FraudClassifier
from src.models.anomaly_detector import AnomalyDetector
from src.features.feature_engineering import FeatureEngineer

def evaluate_dataset(data_csv: str):
    if not os.path.exists(data_csv):
        print(f"Dataset not found at {data_csv}")
        return

    print(f"\n======================================")
    print(f"Evaluating Dataset: {data_csv}")
    print(f"======================================")
    df = pd.read_csv(data_csv)
    
    # Check if raw columns exist or if we can just use the features directly
    engineer = FeatureEngineer()
    try:
        X = engineer.transform_dataframe(df)
    except Exception as e:
        print("Could not transform using FeatureEngineer. Attempting to use columns directly...")
        from src.features.feature_engineering import FEATURE_COLUMNS
        X = df[FEATURE_COLUMNS]
        
    if "is_fraud" not in df.columns:
        print("No 'is_fraud' column found. Cannot evaluate supervised metrics.")
        return
        
    y = df["is_fraud"].values

    print(f"Dataset size: {len(X)} samples")
    print(f"Fraud rate: {y.mean():.1%}")

    # 1. Evaluate Fraud Classifier
    print("\n--- Fraud Classifier (Supervised) ---")
    fraud_model = FraudClassifier().load("models/fraud_model.pkl")
    
    probas = fraud_model.model.predict_proba(X)[:, 1]
    preds = (probas > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y, probas)
        print(f"ROC-AUC: {auc:.4f}\n")
    except ValueError:
        print("ROC-AUC not defined for single-class dataset.")
        
    print("Classification Report:")
    print(classification_report(y, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))

    # 2. Evaluate Anomaly Detector
    print("\n--- Anomaly Detector (Unsupervised) ---")
    anomaly_model = AnomalyDetector().load("models/anomaly_model.pkl")
    
    X_scaled = anomaly_model.feature_scaler.transform(X)
    raw_scores = anomaly_model.model.decision_function(X_scaled)
    inverted = -raw_scores.reshape(-1, 1)
    normalized = anomaly_model.score_scaler.transform(inverted).flatten()
    
    avg_legit_score = np.mean(normalized[y == 0]) if sum(y==0) > 0 else 0
    avg_fraud_score = np.mean(normalized[y == 1]) if sum(y==1) > 0 else 0
    
    print(f"Average Anomaly Score for Legitimate: {avg_legit_score:.4f}")
    print(f"Average Anomaly Score for Fraudulent: {avg_fraud_score:.4f}")
    
    if avg_fraud_score > avg_legit_score:
        print("-> Success! Fraudulent transactions are generally scoring higher as anomalies.")
    else:
        print("-> Warning! Anomaly scores do not align with known fraud.")


if __name__ == "__main__":
    evaluate_dataset("training_DS_1.csv")
    evaluate_dataset("training_DS_2.csv")

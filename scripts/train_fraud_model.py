"""Train the supervised Gradient Boosting fraud classifier model with tuning and SMOTE."""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

from src.features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.models.fraud_classifier import FraudClassifier
from scripts.generate_demo_data import generate_synthetic_training_data


def train_fraud_model(
    data_csv: str = "data/raw/train_transactions.csv",
    model_output: str = "models/fraud_model.pkl",
) -> FraudClassifier:
    """Train and persist the Fraud Classifier using SMOTE and GridSearchCV.

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

    print("Extracting feature matrix...")
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        feature_engineer = FeatureEngineer()
        X = feature_engineer.transform_dataframe(df)
    else:
        print("Features already present in dataset. Bypassing FeatureEngineer...")
        X = df[FEATURE_COLUMNS]
        
    y = df["is_fraud"].values

    print(f"Original training set: {len(X)} samples with {len(FEATURE_COLUMNS)} features. Fraud rate: {y.mean():.1%}")

    # 1. Handle Class Imbalance with SMOTE
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Resampled training set: {len(X_resampled)} samples. Fraud rate: {y_resampled.mean():.1%}")

    # 2. Hyperparameter Tuning with GridSearchCV
    print("Running GridSearchCV for Hyperparameter Tuning (this may take a few seconds)...")
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_iter': [50, 100, 200],
    }
    
    base_estimator = HistGradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X_resampled, y_resampled)
    
    best_params = grid_search.best_params_
    print(f"Best hyperparameters found: {best_params}")

    # 3. Train final wrapper model
    classifier = FraudClassifier(
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter'],
        model_path=model_output
    )
    # the train method will re-fit using the best parameters
    classifier.train(X_resampled, y_resampled)

    # In-sample baseline diagnostic (prototype verification) on original un-sampled data
    probas = classifier.model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probas)
    print(f"Model fitted successfully. Training ROC-AUC on original data: {auc:.4f}")

    saved_path = classifier.save(model_output)
    print(f"Fraud model saved to: {saved_path}")

    return classifier


if __name__ == "__main__":
    train_fraud_model()

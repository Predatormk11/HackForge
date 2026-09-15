"""Preprocessing routines for raw transactions and datasets."""
import os
from typing import Any, Dict
import pandas as pd

from src.schemas.transaction import TransactionInput


def preprocess_transaction(tx: TransactionInput) -> Dict[str, Any]:
    """Clean and standardize a single TransactionInput instance into a dictionary.

    Args:
        tx: Validated TransactionInput instance.

    Returns:
        Dictionary of standardized transaction values with missing values filled with defaults.
    """
    timestamp = tx.timestamp

    return {
        "transaction_id": tx.transaction_id,
        "user_id": tx.user_id,
        "amount": float(tx.amount),
        "transaction_type": tx.transaction_type.upper(),
        "timestamp": timestamp,
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "merchant_id": tx.merchant_id,
        "device_id": tx.device_id,
        "location": tx.location,
        "account_age_days": float(max(0, tx.account_age_days)),
        "previous_transaction_count": float(max(0, tx.previous_transaction_count)),
        "previous_average_amount": float(max(0.0, tx.previous_average_amount)),
        "failed_attempts": float(max(0, tx.failed_attempts)),
        "previous_device_known": 1.0 if tx.previous_device_known else 0.0,
        "previous_merchant_known": 1.0 if tx.previous_merchant_known else 0.0,
        "transactions_last_24h": float(max(1, tx.transactions_last_24h)),
        "is_location_consistent": 1.0 if tx.is_location_consistent else 0.0,
    }


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load transaction records from a CSV file.

    Args:
        file_path: Path to CSV dataset.

    Returns:
        Pandas DataFrame of transaction records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

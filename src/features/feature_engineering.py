"""Feature engineering pipeline for transaction risk scoring."""
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from src.schemas.transaction import TransactionInput
from src.data.preprocessing import preprocess_transaction

# Deterministic ordered list of numerical feature columns used by models
FEATURE_COLUMNS: List[str] = [
    "amount",
    "amount_deviation",
    "amount_to_avg_ratio",
    "transaction_velocity",
    "new_device_indicator",
    "new_merchant_indicator",
    "unusual_time_indicator",
    "location_deviation_indicator",
    "failed_attempts",
    "account_age_days",
    "previous_transaction_count",
    "previous_average_amount",
    "hour_of_day",
    "day_of_week",
]


class FeatureEngineer:
    """Deterministic feature extractor for financial transactions."""

    def __init__(self, unusual_hours: List[int] = None):
        """Initialize feature engineer with configurable parameters.

        Args:
            unusual_hours: List of integer hours (0-23) considered off-peak/unusual.
        """
        self.unusual_hours = unusual_hours if unusual_hours is not None else [0, 1, 2, 3, 4, 5]

    def extract_features_dict(self, tx: Union[TransactionInput, Dict[str, Any]]) -> Dict[str, float]:
        """Extract a dictionary of numerical feature values from a transaction.

        Args:
            tx: TransactionInput model instance or raw dictionary.

        Returns:
            Dictionary mapping feature column names to float values.
        """
        if isinstance(tx, TransactionInput):
            clean_data = preprocess_transaction(tx)
        else:
            # If already dict or raw dict, validate and preprocess
            from src.data.validation import validate_transaction_data
            model = validate_transaction_data(tx)
            clean_data = preprocess_transaction(model)

        amount = float(clean_data["amount"])
        prev_avg = float(clean_data["previous_average_amount"])
        hour = int(clean_data["hour_of_day"])
        day = int(clean_data["day_of_week"])
        velocity = float(clean_data["transactions_last_24h"])

        # Feature calculations
        # Amount deviation: relative deviation (amount - avg) / max(avg, 1.0)
        safe_avg = max(prev_avg, 1.0)
        amount_deviation = (amount - prev_avg) / safe_avg
        amount_to_avg_ratio = amount / safe_avg

        # Indicators (1.0 = risk present, 0.0 = normal)
        new_device_indicator = 1.0 if clean_data["previous_device_known"] == 0.0 else 0.0
        new_merchant_indicator = 1.0 if clean_data["previous_merchant_known"] == 0.0 else 0.0
        unusual_time_indicator = 1.0 if hour in self.unusual_hours else 0.0
        location_deviation_indicator = 1.0 if clean_data["is_location_consistent"] == 0.0 else 0.0

        return {
            "amount": amount,
            "amount_deviation": float(amount_deviation),
            "amount_to_avg_ratio": float(amount_to_avg_ratio),
            "transaction_velocity": velocity,
            "new_device_indicator": new_device_indicator,
            "new_merchant_indicator": new_merchant_indicator,
            "unusual_time_indicator": unusual_time_indicator,
            "location_deviation_indicator": location_deviation_indicator,
            "failed_attempts": float(clean_data["failed_attempts"]),
            "account_age_days": float(clean_data["account_age_days"]),
            "previous_transaction_count": float(clean_data["previous_transaction_count"]),
            "previous_average_amount": prev_avg,
            "hour_of_day": float(hour),
            "day_of_week": float(day),
        }

    def transform_single(self, tx: Union[TransactionInput, Dict[str, Any]]) -> pd.DataFrame:
        """Transform a single transaction into a single-row DataFrame aligned with FEATURE_COLUMNS.

        Args:
            tx: TransactionInput or dictionary.

        Returns:
            pd.DataFrame with 1 row and columns matching FEATURE_COLUMNS.
        """
        feats = self.extract_features_dict(tx)
        return pd.DataFrame([feats], columns=FEATURE_COLUMNS)

    def transform_batch(self, transactions: List[Union[TransactionInput, Dict[str, Any]]]) -> pd.DataFrame:
        """Transform multiple transactions into a DataFrame aligned with FEATURE_COLUMNS.

        Args:
            transactions: List of TransactionInput or dictionaries.

        Returns:
            pd.DataFrame with N rows and columns matching FEATURE_COLUMNS.
        """
        feature_dicts = [self.extract_features_dict(tx) for tx in transactions]
        return pd.DataFrame(feature_dicts, columns=FEATURE_COLUMNS)

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a pandas DataFrame of raw transaction rows into feature DataFrame.

        Args:
            df: DataFrame containing raw transaction columns.

        Returns:
            DataFrame with FEATURE_COLUMNS.
        """
        records = df.to_dict(orient="records")
        return self.transform_batch(records)

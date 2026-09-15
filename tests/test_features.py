"""Unit tests for transaction data validation and feature engineering pipeline."""
from datetime import datetime
import pytest
from pydantic import ValidationError

from src.data.validation import validate_transaction_data
from src.features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.schemas.transaction import TransactionInput


def test_transaction_validation_success():
    """Test valid transaction creation."""
    payload = {
        "transaction_id": "TX_VALID_001",
        "user_id": "USR_TEST_01",
        "amount": 150.75,
        "transaction_type": "PURCHASE",
        "timestamp": "2026-09-15T12:00:00",
        "merchant_id": "MERCH_AMAZON",
        "device_id": "DEV_CHROME_WIN",
        "location": "New York, US",
    }
    tx = validate_transaction_data(payload)
    assert tx.transaction_id == "TX_VALID_001"
    assert tx.amount == 150.75
    assert tx.account_age_days == 30  # Default value
    assert tx.previous_device_known is True  # Default value


def test_transaction_validation_negative_amount():
    """Test rejection of negative monetary amount."""
    with pytest.raises(ValueError):
        validate_transaction_data({
            "transaction_id": "TX_INVALID_002",
            "user_id": "USR_TEST_01",
            "amount": -50.0,
            "transaction_type": "PURCHASE",
            "timestamp": "2026-09-15T12:00:00",
            "merchant_id": "MERCH_AMAZON",
            "device_id": "DEV_01",
            "location": "NY",
        })


def test_transaction_validation_empty_string():
    """Test rejection of empty string fields."""
    with pytest.raises(ValueError):
        validate_transaction_data({
            "transaction_id": "   ",
            "user_id": "USR_01",
            "amount": 50.0,
            "transaction_type": "PURCHASE",
            "timestamp": "2026-09-15T12:00:00",
            "merchant_id": "MERCH_01",
            "device_id": "DEV_01",
            "location": "NY",
        })


def test_feature_engineering_extraction():
    """Test deterministic feature vector calculation."""
    fe = FeatureEngineer(unusual_hours=[0, 1, 2, 3, 4, 5])

    tx = TransactionInput(
        transaction_id="TX_FEAT_001",
        user_id="USR_FEAT_01",
        amount=300.0,
        transaction_type="PURCHASE",
        timestamp=datetime(2026, 9, 15, 3, 30, 0),  # 3:30 AM -> unusual hour
        merchant_id="MERCH_NEW_99",
        device_id="DEV_NEW_99",
        location="Unknown_Location",
        account_age_days=10,
        previous_transaction_count=5,
        previous_average_amount=100.0,
        failed_attempts=2,
        previous_device_known=False,
        previous_merchant_known=False,
        transactions_last_24h=6,
        is_location_consistent=False,
    )

    feat_dict = fe.extract_features_dict(tx)
    assert feat_dict["amount"] == 300.0
    assert feat_dict["amount_deviation"] == 2.0  # (300 - 100) / 100
    assert feat_dict["amount_to_avg_ratio"] == 3.0
    assert feat_dict["new_device_indicator"] == 1.0
    assert feat_dict["new_merchant_indicator"] == 1.0
    assert feat_dict["unusual_time_indicator"] == 1.0  # 3 AM is unusual
    assert feat_dict["location_deviation_indicator"] == 1.0
    assert feat_dict["failed_attempts"] == 2.0
    assert feat_dict["transaction_velocity"] == 6.0

    # Test DataFrame transformation columns
    df = fe.transform_single(tx)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1

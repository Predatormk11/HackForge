"""Unit tests for the deterministic Rule Engine."""
from datetime import datetime
from src.rules.rule_engine import RuleEngine
from src.schemas.transaction import TransactionInput


def test_rule_clean_transaction_no_triggers():
    """Test that a clean normal transaction triggers zero rules and scores 0.0."""
    engine = RuleEngine()
    tx = TransactionInput(
        transaction_id="TX_CLEAN_001",
        user_id="USR_01",
        amount=50.0,
        transaction_type="PURCHASE",
        timestamp=datetime(2026, 9, 15, 14, 0, 0),  # 2 PM
        merchant_id="MERCH_01",
        device_id="DEV_01",
        location="New York, US",
        previous_average_amount=50.0,
        previous_device_known=True,
        previous_merchant_known=True,
        transactions_last_24h=1,
        is_location_consistent=True,
        failed_attempts=0,
    )

    result = engine.evaluate(tx)
    assert result.rule_score == 0.0
    assert len(result.triggered_rules) == 0
    assert len(result.reasons) == 0


def test_rule_high_amount_deviation():
    """Test HIGH_AMOUNT_DEVIATION rule trigger."""
    engine = RuleEngine()
    tx = TransactionInput(
        transaction_id="TX_AMT_001",
        user_id="USR_01",
        amount=500.0,  # 5x previous average (100.0) -> > 3.0 threshold
        transaction_type="PURCHASE",
        timestamp=datetime(2026, 9, 15, 14, 0, 0),
        merchant_id="MERCH_01",
        device_id="DEV_01",
        location="New York, US",
        previous_average_amount=100.0,
    )

    result = engine.evaluate(tx)
    assert "HIGH_AMOUNT_DEVIATION" in result.triggered_rules
    assert result.rule_score > 0.0


def test_rule_all_seven_triggers():
    """Test that all 7 rules trigger when all conditions are met and score sums/clamps properly."""
    engine = RuleEngine()
    tx = TransactionInput(
        transaction_id="TX_ALL_001",
        user_id="USR_01",
        amount=2000.0,
        transaction_type="TRANSFER",
        timestamp=datetime(2026, 9, 15, 2, 0, 0),  # UNUSUAL_TIME (2 AM)
        merchant_id="MERCH_NEW",
        device_id="DEV_NEW",
        location="Offshore",
        previous_average_amount=50.0,  # HIGH_AMOUNT_DEVIATION
        previous_device_known=False,  # NEW_DEVICE
        previous_merchant_known=False,  # NEW_MERCHANT
        transactions_last_24h=10,  # HIGH_TRANSACTION_VELOCITY
        is_location_consistent=False,  # LOCATION_DEVIATION
        failed_attempts=5,  # MULTIPLE_FAILED_ATTEMPTS
    )

    result = engine.evaluate(tx)
    expected_rules = [
        "HIGH_AMOUNT_DEVIATION",
        "NEW_DEVICE",
        "UNUSUAL_TIME",
        "HIGH_TRANSACTION_VELOCITY",
        "NEW_MERCHANT",
        "LOCATION_DEVIATION",
        "MULTIPLE_FAILED_ATTEMPTS",
    ]
    for rule in expected_rules:
        assert rule in result.triggered_rules

    assert 0.0 <= result.rule_score <= 1.0
    assert len(result.reasons) == 7

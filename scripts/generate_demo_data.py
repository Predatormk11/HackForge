"""Synthetic data generator for training baseline models and providing demo scenarios."""
import json
import os
from datetime import datetime, timedelta
import random
from typing import Dict
import pandas as pd
import numpy as np


def generate_synthetic_training_data(n_samples: int = 1200, output_csv: str = "data/raw/train_transactions.csv") -> pd.DataFrame:
    """Generate a realistic dataset of normal and fraudulent transactions for baseline model training.

    Args:
        n_samples: Total number of transaction rows to synthesize.
        output_csv: Target CSV path.

    Returns:
        Generated pandas DataFrame.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    records = []
    base_time = datetime(2026, 9, 1, 0, 0, 0)

    # 90% legitimate, 10% fraudulent/anomalous
    n_fraud = int(n_samples * 0.12)
    n_legit = n_samples - n_fraud

    user_ids = [f"USR_{i:04d}" for i in range(1, 101)]
    merchants = [f"MERCH_{i:03d}" for i in range(1, 51)]
    devices = [f"DEV_{i:04d}" for i in range(1, 150)]
    locations = ["New York, US", "San Francisco, US", "London, UK", "Berlin, DE", "Tokyo, JP", "Sydney, AU"]
    foreign_locations = ["Lagos, NG", "Bucharest, RO", "Unknown_IP_Proxy", "Cayman_Islands"]

    # 1. Normal transactions
    for i in range(n_legit):
        tx_id = f"TX_NORM_{i+1:05d}"
        u_id = random.choice(user_ids)
        prev_avg = random.uniform(30.0, 150.0)
        # Normal amounts around previous average
        amount = max(5.0, np.random.normal(prev_avg, prev_avg * 0.35))
        # Normal daytime hours (7 AM to 11 PM)
        hour = random.randint(7, 23)
        day_offset = random.randint(0, 14)
        minute = random.randint(0, 59)
        ts = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)

        records.append({
            "transaction_id": tx_id,
            "user_id": u_id,
            "amount": round(amount, 2),
            "transaction_type": random.choice(["PURCHASE", "PURCHASE", "PAYMENT", "TRANSFER"]),
            "timestamp": ts.isoformat(),
            "merchant_id": random.choice(merchants[:30]),
            "device_id": random.choice(devices[:80]),
            "location": random.choice(locations),
            "account_age_days": random.randint(60, 1200),
            "previous_transaction_count": random.randint(15, 300),
            "previous_average_amount": round(prev_avg, 2),
            "failed_attempts": 0 if random.random() > 0.05 else 1,
            "previous_device_known": True if random.random() > 0.05 else False,
            "previous_merchant_known": True if random.random() > 0.10 else False,
            "transactions_last_24h": random.randint(1, 3),
            "is_location_consistent": True if random.random() > 0.03 else False,
            "is_fraud": 0,
        })

    # 2. Fraudulent / Anomalous transactions
    for i in range(n_fraud):
        tx_id = f"TX_FRAUD_{i+1:05d}"
        u_id = random.choice(user_ids)
        prev_avg = random.uniform(30.0, 100.0)
        # Big spike in amount
        amount = max(prev_avg * random.uniform(4.0, 15.0), random.uniform(1200.0, 6500.0))
        # Off-peak unusual hours (0-5 AM)
        hour = random.choice([0, 1, 2, 3, 4, 5])
        day_offset = random.randint(0, 14)
        minute = random.randint(0, 59)
        ts = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)

        records.append({
            "transaction_id": tx_id,
            "user_id": u_id,
            "amount": round(amount, 2),
            "transaction_type": random.choice(["TRANSFER", "WITHDRAWAL", "PURCHASE"]),
            "timestamp": ts.isoformat(),
            "merchant_id": random.choice(merchants[30:]),
            "device_id": f"DEV_NEW_{random.randint(900, 999)}",
            "location": random.choice(foreign_locations),
            "account_age_days": random.randint(1, 45),
            "previous_transaction_count": random.randint(1, 10),
            "previous_average_amount": round(prev_avg, 2),
            "failed_attempts": random.choice([2, 3, 4, 5]),
            "previous_device_known": False,
            "previous_merchant_known": False,
            "transactions_last_24h": random.randint(5, 12),
            "is_location_consistent": False,
            "is_fraud": 1,
        })

    random.shuffle(records)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} synthetic training records saved to {output_csv}")
    return df


def generate_demo_scenarios(output_json: str = "data/sample/transactions.json") -> Dict[str, dict]:
    """Generate 3 predefined benchmark demo scenarios for the Streamlit UI and API testing.

    Returns:
        Dictionary mapping scenario keys to transaction payload dictionaries.
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    scenarios = {
        "scenario_1_normal": {
            "name": "Scenario 1: Legitimate Everyday Purchase",
            "description": "Routine $45.00 grocery purchase from a known device and recognized merchant during standard afternoon hours with zero failed attempts.",
            "expected_level": "LOW",
            "expected_action": "APPROVE",
            "data": {
                "transaction_id": "TX_DEMO_001",
                "user_id": "USR_ALICE_101",
                "amount": 45.00,
                "transaction_type": "PURCHASE",
                "timestamp": "2026-09-15T14:30:00",
                "merchant_id": "MERCH_GROCERY_042",
                "device_id": "DEV_IPHONE_14_A",
                "location": "Seattle, US",
                "account_age_days": 420,
                "previous_transaction_count": 85,
                "previous_average_amount": 52.00,
                "failed_attempts": 0,
                "previous_device_known": True,
                "previous_merchant_known": True,
                "transactions_last_24h": 1,
                "is_location_consistent": True,
            },
        },
        "scenario_2_suspicious": {
            "name": "Scenario 2: Suspicious New Device & High Value",
            "description": "Uncharacteristic $850.00 electronics purchase from an unrecognized device at 3:15 AM with an unfamiliar merchant.",
            "expected_level": "MEDIUM",
            "expected_action": "ADDITIONAL_VERIFICATION",
            "data": {
                "transaction_id": "TX_DEMO_002",
                "user_id": "USR_BOB_202",
                "amount": 850.00,
                "transaction_type": "PURCHASE",
                "timestamp": "2026-09-15T03:15:00",
                "merchant_id": "MERCH_LUXURY_999",
                "device_id": "DEV_UNRECOGNIZED_X1",
                "location": "Chicago, US",
                "account_age_days": 180,
                "previous_transaction_count": 40,
                "previous_average_amount": 95.00,
                "failed_attempts": 1,
                "previous_device_known": False,
                "previous_merchant_known": False,
                "transactions_last_24h": 3,
                "is_location_consistent": True,
            },
        },
        "scenario_3_critical": {
            "name": "Scenario 3: Highly Suspicious Account Takeover & Rapid Drain",
            "description": "Extreme $4,950.00 wire transfer from an unrecognized foreign IP proxy at 2:45 AM following 4 failed password attempts and burst transaction velocity.",
            "expected_level": "CRITICAL",
            "expected_action": "BLOCK_AND_INVESTIGATE",
            "data": {
                "transaction_id": "TX_DEMO_003",
                "user_id": "USR_CHARLIE_303",
                "amount": 4950.00,
                "transaction_type": "TRANSFER",
                "timestamp": "2026-09-15T02:45:00",
                "merchant_id": "MERCH_CRYPTO_OFFSHORE",
                "device_id": "DEV_TOR_EXIT_NODE_9",
                "location": "Unknown_Proxy_Lagos",
                "account_age_days": 15,
                "previous_transaction_count": 3,
                "previous_average_amount": 60.00,
                "failed_attempts": 4,
                "previous_device_known": False,
                "previous_merchant_known": False,
                "transactions_last_24h": 8,
                "is_location_consistent": False,
            },
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Generated sample scenarios saved to {output_json}")
    return scenarios


if __name__ == "__main__":
    generate_synthetic_training_data()
    generate_demo_scenarios()

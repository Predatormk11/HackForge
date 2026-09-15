"""Integration tests for Agent Orchestrator and FastAPI service."""
from datetime import datetime
from fastapi.testclient import TestClient

from api.main import app
from src.agent.orchestrator import AgentOrchestrator
from src.schemas.transaction import TransactionInput, TransactionRiskOutput, RiskLevel


def test_agent_orchestrator_normal_transaction():
    """Test full pipeline execution on a standard low-risk transaction."""
    orchestrator = AgentOrchestrator(config_path="config/config.yaml")

    tx = TransactionInput(
        transaction_id="TX_TEST_001",
        user_id="USR_ALICE",
        amount=35.0,
        transaction_type="PURCHASE",
        timestamp=datetime(2026, 9, 15, 13, 30, 0),
        merchant_id="MERCH_SAFE",
        device_id="DEV_SAFE",
        location="New York, US",
        account_age_days=300,
        previous_transaction_count=50,
        previous_average_amount=40.0,
        failed_attempts=0,
        previous_device_known=True,
        previous_merchant_known=True,
        transactions_last_24h=1,
        is_location_consistent=True,
    )

    result = orchestrator.analyze_transaction(tx)

    assert isinstance(result, TransactionRiskOutput)
    assert result.transaction_id == "TX_TEST_001"
    assert 0.0 <= result.fraud_probability <= 1.0
    assert 0.0 <= result.anomaly_score <= 1.0
    assert 0.0 <= result.rule_score <= 1.0
    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert len(result.risk_factors) > 0


def test_agent_orchestrator_high_risk_transaction():
    """Test full pipeline execution on a severe risk transaction."""
    orchestrator = AgentOrchestrator(config_path="config/config.yaml")

    tx = TransactionInput(
        transaction_id="TX_TEST_CRITICAL",
        user_id="USR_CHARLIE",
        amount=4800.0,
        transaction_type="TRANSFER",
        timestamp=datetime(2026, 9, 15, 3, 10, 0),
        merchant_id="MERCH_RISKY",
        device_id="DEV_UNKNOWN",
        location="Foreign_Proxy",
        account_age_days=5,
        previous_transaction_count=2,
        previous_average_amount=50.0,
        failed_attempts=4,
        previous_device_known=False,
        previous_merchant_known=False,
        transactions_last_24h=9,
        is_location_consistent=False,
    )

    result = orchestrator.analyze_transaction(tx)
    assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert len(result.risk_factors) >= 3


def test_fastapi_endpoints():
    """Test FastAPI /health, /api/v1/analyze, and /api/v1/scenarios endpoints."""
    with TestClient(app) as client:
        # Health check
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json() == {"status": "ok"}

        # Scenarios
        resp_scenarios = client.get("/api/v1/scenarios")
        assert resp_scenarios.status_code == 200
        assert "scenario_1_normal" in resp_scenarios.json()

        # Analyze transaction
        sample_payload = {
            "transaction_id": "TX_API_001",
            "user_id": "USR_API_USER",
            "amount": 75.50,
            "transaction_type": "PURCHASE",
            "timestamp": "2026-09-15T15:00:00",
            "merchant_id": "MERCH_101",
            "device_id": "DEV_101",
            "location": "Boston, US",
            "previous_average_amount": 70.0,
            "previous_device_known": True,
            "previous_merchant_known": True,
        }
        resp_analyze = client.post("/api/v1/analyze", json=sample_payload)
        assert resp_analyze.status_code == 200
        data = resp_analyze.json()
        assert data["transaction_id"] == "TX_API_001"
        assert "fraud_probability" in data
        assert "anomaly_score" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "risk_factors" in data
        assert "recommended_action" in data

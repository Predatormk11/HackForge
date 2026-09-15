"""Unit tests for risk fusion calculation and decision classification."""
from src.risk.risk_fusion import RiskFusionEngine
from src.risk.decision_engine import DecisionEngine
from src.schemas.transaction import RiskLevel, ActionType


def test_risk_fusion_formula():
    """Test standard default weights in risk fusion (0.60, 0.25, 0.15)."""
    fusion = RiskFusionEngine(fraud_weight=0.60, anomaly_weight=0.25, rule_weight=0.15)

    # All zeros
    assert fusion.calculate(0.0, 0.0, 0.0) == 0.0

    # All ones
    assert fusion.calculate(1.0, 1.0, 1.0) == 1.0

    # Specific calculation: 0.60*0.5 + 0.25*0.4 + 0.15*0.2 = 0.30 + 0.10 + 0.03 = 0.43
    score = fusion.calculate(0.5, 0.4, 0.2)
    assert round(score, 4) == 0.4300


def test_risk_fusion_clamping():
    """Test that out-of-range inputs are clamped between 0 and 1."""
    fusion = RiskFusionEngine()
    assert fusion.calculate(-0.5, 0.0, 0.0) == 0.0
    assert fusion.calculate(1.5, 2.0, 1.2) == 1.0


def test_decision_engine_classification_and_actions():
    """Test threshold tiers and corresponding action mappings."""
    engine = DecisionEngine(low_threshold=0.30, medium_threshold=0.70, high_threshold=0.90)

    # 1. LOW: [0, 0.30) -> APPROVE
    assert engine.classify(0.15) == RiskLevel.LOW
    assert engine.recommend_action(RiskLevel.LOW) == ActionType.APPROVE

    # 2. MEDIUM: [0.30, 0.70) -> ADDITIONAL_VERIFICATION
    assert engine.classify(0.30) == RiskLevel.MEDIUM
    assert engine.classify(0.55) == RiskLevel.MEDIUM
    assert engine.recommend_action(RiskLevel.MEDIUM) == ActionType.ADDITIONAL_VERIFICATION

    # 3. HIGH: [0.70, 0.90) -> HOLD_AND_REVIEW
    assert engine.classify(0.70) == RiskLevel.HIGH
    assert engine.classify(0.85) == RiskLevel.HIGH
    assert engine.recommend_action(RiskLevel.HIGH) == ActionType.HOLD_AND_REVIEW

    # 4. CRITICAL: [0.90, 1.00] -> BLOCK_AND_INVESTIGATE
    assert engine.classify(0.90) == RiskLevel.CRITICAL
    assert engine.classify(0.98) == RiskLevel.CRITICAL
    assert engine.recommend_action(RiskLevel.CRITICAL) == ActionType.BLOCK_AND_INVESTIGATE

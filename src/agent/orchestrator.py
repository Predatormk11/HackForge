"""Fraud Risk Agent Orchestrator coordinating all analytical engines."""
import os
from typing import Any, Dict, Optional, Union
import yaml

from src.agent.explanation import ExplanationGenerator
from src.data.validation import validate_transaction_data
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_detector import AnomalyDetector
from src.models.fraud_classifier import FraudClassifier
from src.risk.decision_engine import DecisionEngine
from src.risk.risk_fusion import RiskFusionEngine
from src.rules.rule_engine import RuleEngine
from src.schemas.transaction import TransactionInput, TransactionRiskOutput


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dictionary of configuration options.
    """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class AgentOrchestrator:
    """Central orchestrator for the Fraud Detection & Transaction Risk Agent."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialize orchestrator and sub-analytical modules.

        Args:
            config_path: Path to config YAML file.
            config_dict: Optional direct configuration dictionary.
        """
        self.config = config_dict if config_dict is not None else load_config(config_path)

        # 1. Feature Engineering
        unusual_hours = self.config.get("rules", {}).get("unusual_hours", [0, 1, 2, 3, 4, 5])
        self.feature_engineer = FeatureEngineer(unusual_hours=unusual_hours)

        # 2. Model paths
        models_cfg = self.config.get("models", {})
        fraud_model_path = models_cfg.get("fraud_model_path", "models/fraud_model.pkl")
        anomaly_model_path = models_cfg.get("anomaly_model_path", "models/anomaly_model.pkl")

        # 3. Models
        self.fraud_classifier = FraudClassifier(model_path=fraud_model_path)
        self.anomaly_detector = AnomalyDetector(model_path=anomaly_model_path)

        # 4. Analytical Engines
        self.rule_engine = RuleEngine(config=self.config)
        self.risk_fusion = RiskFusionEngine(config=self.config)
        self.decision_engine = DecisionEngine(config=self.config)
        self.explanation_generator = ExplanationGenerator()

    def analyze_transaction(
        self,
        transaction: Union[TransactionInput, Dict[str, Any]],
    ) -> TransactionRiskOutput:
        """Execute full end-to-end risk evaluation workflow for a transaction.

        Workflow:
        1. Validate input schema
        2. Extract numerical features
        3. Predict fraud probability (Supervised)
        4. Predict anomaly score (Unsupervised)
        5. Evaluate deterministic risk rules
        6. Fuse risk scores
        7. Classify into risk level
        8. Determine recommended action
        9. Generate explainable risk factors
        10. Return TransactionRiskOutput

        Args:
            transaction: Raw transaction dictionary or TransactionInput model instance.

        Returns:
            TransactionRiskOutput containing all scores, tier, action, and explanations.
        """
        # 1. Validate Input
        tx: TransactionInput = validate_transaction_data(transaction)

        # 2. Generate Features
        feature_df = self.feature_engineer.transform_single(tx)
        feature_dict = self.feature_engineer.extract_features_dict(tx)

        # 3. Supervised Fraud Classification
        fraud_prob = self.fraud_classifier.predict_probability(feature_df)

        # 4. Unsupervised Anomaly Detection
        anomaly_score = self.anomaly_detector.predict_anomaly_score(feature_df)

        # 5. Rule Engine Evaluation
        rule_result = self.rule_engine.evaluate(tx, features=feature_dict)

        # 6. Risk Fusion
        risk_score = self.risk_fusion.calculate(
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_score,
            rule_score=rule_result.rule_score,
        )

        # 7. Risk Level Classification
        risk_level = self.decision_engine.classify(risk_score)

        # 8. Action Recommendation
        recommended_action = self.decision_engine.recommend_action(risk_level)

        # 9. Explanation Generation
        risk_factors = self.explanation_generator.generate_explanations(
            transaction=tx,
            rule_result=rule_result,
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
        )

        # 10. Final Output
        return TransactionRiskOutput(
            transaction_id=tx.transaction_id,
            fraud_probability=round(fraud_prob, 4),
            anomaly_score=round(anomaly_score, 4),
            rule_score=round(rule_result.rule_score, 4),
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
        )

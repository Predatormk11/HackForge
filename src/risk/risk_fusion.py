"""Risk fusion engine combining supervised fraud probability, anomaly score, and rule signals."""
from typing import Any, Dict, Optional
import numpy as np


class RiskFusionEngine:
    """Combines heterogeneous risk signals into a single unified risk score."""

    def __init__(
        self,
        fraud_weight: float = 0.60,
        anomaly_weight: float = 0.25,
        rule_weight: float = 0.15,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the risk fusion engine.

        Args:
            fraud_weight: Weight given to supervised fraud model probability.
            anomaly_weight: Weight given to unsupervised anomaly score.
            rule_weight: Weight given to deterministic rule engine score.
            config: Optional config dictionary to override weights.
        """
        if config and "risk" in config:
            risk_cfg = config["risk"]
            self.fraud_weight = float(risk_cfg.get("fraud_weight", fraud_weight))
            self.anomaly_weight = float(risk_cfg.get("anomaly_weight", anomaly_weight))
            self.rule_weight = float(risk_cfg.get("rule_weight", rule_weight))
        else:
            self.fraud_weight = float(fraud_weight)
            self.anomaly_weight = float(anomaly_weight)
            self.rule_weight = float(rule_weight)

        # Normalize weights so they sum to 1.0
        total_weight = self.fraud_weight + self.anomaly_weight + self.rule_weight
        if total_weight > 0:
            self.fraud_weight /= total_weight
            self.anomaly_weight /= total_weight
            self.rule_weight /= total_weight

    def calculate(
        self,
        fraud_probability: float,
        anomaly_score: float,
        rule_score: float,
    ) -> float:
        """Fuse individual risk components into an aggregate risk score [0, 1].

        Args:
            fraud_probability: Float between 0.0 and 1.0.
            anomaly_score: Float between 0.0 and 1.0.
            rule_score: Float between 0.0 and 1.0.

        Returns:
            Fused risk score clamped between 0.0 and 1.0.
        """
        f_prob = float(np.clip(fraud_probability, 0.0, 1.0))
        a_score = float(np.clip(anomaly_score, 0.0, 1.0))
        r_score = float(np.clip(rule_score, 0.0, 1.0))

        raw_fused = (
            (self.fraud_weight * f_prob)
            + (self.anomaly_weight * a_score)
            + (self.rule_weight * r_score)
        )

        return float(np.clip(raw_fused, 0.0, 1.0))

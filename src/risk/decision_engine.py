"""Decision engine mapping continuous risk scores to risk tiers and recommended actions."""
from typing import Any, Dict, Optional, Union

from src.schemas.transaction import ActionType, RiskLevel


class DecisionEngine:
    """Classifies risk scores into risk levels and determines mitigation actions."""

    def __init__(
        self,
        low_threshold: float = 0.30,
        medium_threshold: float = 0.70,
        high_threshold: float = 0.90,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize decision engine thresholds.

        Args:
            low_threshold: Upper bound for LOW risk (e.g. < 0.30).
            medium_threshold: Upper bound for MEDIUM risk (e.g. < 0.70).
            high_threshold: Upper bound for HIGH risk (e.g. < 0.90). Scores >= high_threshold are CRITICAL.
            config: Optional config dictionary.
        """
        if config and "thresholds" in config:
            thresh_cfg = config["thresholds"]
            self.low_threshold = float(thresh_cfg.get("low", low_threshold))
            self.medium_threshold = float(thresh_cfg.get("medium", medium_threshold))
            self.high_threshold = float(thresh_cfg.get("high", high_threshold))
        else:
            self.low_threshold = float(low_threshold)
            self.medium_threshold = float(medium_threshold)
            self.high_threshold = float(high_threshold)

    def classify(self, risk_score: float) -> RiskLevel:
        """Classify numerical risk score into a categorical RiskLevel.

        Args:
            risk_score: Float between 0.0 and 1.0.

        Returns:
            RiskLevel: LOW, MEDIUM, HIGH, or CRITICAL.
        """
        if risk_score < self.low_threshold:
            return RiskLevel.LOW
        elif risk_score < self.medium_threshold:
            return RiskLevel.MEDIUM
        elif risk_score < self.high_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def recommend_action(self, risk_level: Union[RiskLevel, str]) -> ActionType:
        """Map risk level to a recommended policy action.

        Args:
            risk_level: RiskLevel enum or string.

        Returns:
            ActionType: APPROVE, ADDITIONAL_VERIFICATION, HOLD_AND_REVIEW, or BLOCK_AND_INVESTIGATE.
        """
        level_str = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level).upper()

        if level_str == RiskLevel.LOW.value:
            return ActionType.APPROVE
        elif level_str == RiskLevel.MEDIUM.value:
            return ActionType.ADDITIONAL_VERIFICATION
        elif level_str == RiskLevel.HIGH.value:
            return ActionType.HOLD_AND_REVIEW
        elif level_str == RiskLevel.CRITICAL.value:
            return ActionType.BLOCK_AND_INVESTIGATE
        else:
            return ActionType.ADDITIONAL_VERIFICATION

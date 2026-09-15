"""Evidence-grounded explanation generator for risk assessments."""
from typing import List, Optional

from src.schemas.transaction import RiskLevel, RuleResult, TransactionInput


class ExplanationGenerator:
    """Generates transparent, factual explanations directly grounded in analytical evidence."""

    def generate_explanations(
        self,
        transaction: TransactionInput,
        rule_result: RuleResult,
        fraud_probability: float,
        anomaly_score: float,
        risk_level: RiskLevel,
    ) -> List[str]:
        """Generate human-readable risk factors strictly from observed signals.

        Args:
            transaction: The evaluated transaction input.
            rule_result: Output from the rule engine.
            fraud_probability: Estimated fraud probability from classifier.
            anomaly_score: Anomaly score from anomaly detector.
            risk_level: Final classified risk tier.

        Returns:
            List of clear, non-hallucinated explanation strings.
        """
        explanations: List[str] = []

        # 1. Include factual explanations produced by triggered rules
        for reason in rule_result.reasons:
            if reason not in explanations:
                explanations.append(reason)

        # 2. Add statistical model observations if elevated
        if fraud_probability >= 0.60:
            explanations.append(
                f"Supervised fraud model detected high-risk behavioral pattern (estimated probability: {fraud_probability:.1%})."
            )
        elif fraud_probability >= 0.35:
            explanations.append(
                f"Supervised fraud model detected moderate risk indicators (estimated probability: {fraud_probability:.1%})."
            )

        if anomaly_score >= 0.60:
            explanations.append(
                f"Unsupervised anomaly detector identified significant outlier characteristics (anomaly score: {anomaly_score:.1%})."
            )
        elif anomaly_score >= 0.40:
            explanations.append(
                f"Unsupervised anomaly detector flagged minor deviation from baseline norms (anomaly score: {anomaly_score:.1%})."
            )

        # 3. Default reassuring explanation for low risk with no flags
        if not explanations and risk_level == RiskLevel.LOW:
            explanations.append("All transaction parameters are consistent with normal user activity baseline.")

        return explanations

"""Deterministic, explainable rule engine for evaluating transaction risk signals."""
from typing import Any, Dict, List, Optional
import numpy as np

from src.schemas.transaction import RuleResult, TransactionInput


class RuleEngine:
    """Evaluates deterministic business and risk rules against a transaction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rule engine with rules configuration.

        Args:
            config: Rules configuration dictionary.
        """
        rules_cfg = config.get("rules", {}) if config else {}
        self.high_amount_multiplier = float(rules_cfg.get("high_amount_multiplier", 3.0))
        self.high_amount_absolute_minimum = float(rules_cfg.get("high_amount_absolute_minimum", 1000.0))
        self.unusual_hours = list(rules_cfg.get("unusual_hours", [0, 1, 2, 3, 4, 5]))
        self.high_velocity_threshold = int(rules_cfg.get("high_velocity_threshold", 5))
        self.failed_attempts_threshold = int(rules_cfg.get("failed_attempts_threshold", 3))

        # Default weights for individual rules
        default_weights = {
            "HIGH_AMOUNT_DEVIATION": 0.25,
            "NEW_DEVICE": 0.15,
            "UNUSUAL_TIME": 0.10,
            "HIGH_TRANSACTION_VELOCITY": 0.20,
            "NEW_MERCHANT": 0.10,
            "LOCATION_DEVIATION": 0.10,
            "MULTIPLE_FAILED_ATTEMPTS": 0.10,
        }
        self.weights = rules_cfg.get("weights", default_weights)

    def evaluate(
        self,
        transaction: TransactionInput,
        features: Optional[Dict[str, float]] = None,
    ) -> RuleResult:
        """Evaluate all deterministic rules against the transaction.

        Args:
            transaction: Validated TransactionInput.
            features: Optional dictionary of precomputed features.

        Returns:
            RuleResult containing rule_score, triggered_rules, and human-readable reasons.
        """
        triggered_rules: List[str] = []
        reasons: List[str] = []
        rule_score_accum = 0.0

        amount = float(transaction.amount)
        prev_avg = float(transaction.previous_average_amount)
        hour = transaction.timestamp.hour

        # Rule 1: HIGH_AMOUNT_DEVIATION
        if prev_avg > 0 and amount > (self.high_amount_multiplier * prev_avg):
            triggered_rules.append("HIGH_AMOUNT_DEVIATION")
            reasons.append(
                f"Transaction amount (${amount:,.2f}) is significantly above historical average (${prev_avg:,.2f})."
            )
            rule_score_accum += self.weights.get("HIGH_AMOUNT_DEVIATION", 0.25)
        elif prev_avg == 0 and amount >= self.high_amount_absolute_minimum:
            triggered_rules.append("HIGH_AMOUNT_DEVIATION")
            reasons.append(
                f"High-value first transaction (${amount:,.2f}) exceeds single-transaction baseline."
            )
            rule_score_accum += self.weights.get("HIGH_AMOUNT_DEVIATION", 0.25)

        # Rule 2: NEW_DEVICE
        if not transaction.previous_device_known:
            triggered_rules.append("NEW_DEVICE")
            reasons.append("New device detected.")
            rule_score_accum += self.weights.get("NEW_DEVICE", 0.15)

        # Rule 3: UNUSUAL_TIME
        if hour in self.unusual_hours:
            triggered_rules.append("UNUSUAL_TIME")
            reasons.append(f"Transaction occurred during an unusual time ({hour:02d}:00).")
            rule_score_accum += self.weights.get("UNUSUAL_TIME", 0.10)

        # Rule 4: HIGH_TRANSACTION_VELOCITY
        if transaction.transactions_last_24h >= self.high_velocity_threshold:
            triggered_rules.append("HIGH_TRANSACTION_VELOCITY")
            reasons.append(
                f"High transaction velocity detected ({transaction.transactions_last_24h} transactions in last 24h)."
            )
            rule_score_accum += self.weights.get("HIGH_TRANSACTION_VELOCITY", 0.20)

        # Rule 5: NEW_MERCHANT
        if not transaction.previous_merchant_known:
            triggered_rules.append("NEW_MERCHANT")
            reasons.append("New merchant detected.")
            rule_score_accum += self.weights.get("NEW_MERCHANT", 0.10)

        # Rule 6: LOCATION_DEVIATION
        if not transaction.is_location_consistent:
            triggered_rules.append("LOCATION_DEVIATION")
            reasons.append("Location differs from normal activity.")
            rule_score_accum += self.weights.get("LOCATION_DEVIATION", 0.10)

        # Rule 7: MULTIPLE_FAILED_ATTEMPTS
        if transaction.failed_attempts >= self.failed_attempts_threshold:
            triggered_rules.append("MULTIPLE_FAILED_ATTEMPTS")
            reasons.append(
                f"Multiple failed attempts recorded prior to authorization ({transaction.failed_attempts} failed attempts)."
            )
            rule_score_accum += self.weights.get("MULTIPLE_FAILED_ATTEMPTS", 0.10)

        # Clamp score to [0.0, 1.0]
        final_score = float(np.clip(rule_score_accum, 0.0, 1.0))

        return RuleResult(
            rule_score=final_score,
            triggered_rules=triggered_rules,
            reasons=reasons,
        )

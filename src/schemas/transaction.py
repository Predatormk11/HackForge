"""Transaction data models and output schemas."""
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Enumeration of risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    """Enumeration of recommended mitigation actions."""
    APPROVE = "APPROVE"
    ADDITIONAL_VERIFICATION = "ADDITIONAL_VERIFICATION"
    HOLD_AND_REVIEW = "HOLD_AND_REVIEW"
    BLOCK_AND_INVESTIGATE = "BLOCK_AND_INVESTIGATE"


class TransactionInput(BaseModel):
    """Input payload representing a single financial transaction event."""
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    user_id: str = Field(..., description="Unique identifier for the user / account")
    amount: float = Field(..., ge=0.0, description="Transaction monetary amount")
    transaction_type: str = Field(..., description="Type of transaction, e.g. PURCHASE, TRANSFER, WITHDRAWAL")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of transaction")
    merchant_id: str = Field(..., description="Merchant identifier or recipient ID")
    device_id: str = Field(..., description="Device fingerprint or identifier")
    location: str = Field(..., description="Geographic location (e.g. City, Country or IP geolocation)")

    # Optional historical/contextual fields with sensible defaults for Phase-1 demo
    account_age_days: int = Field(default=30, ge=0, description="Age of the user account in days")
    previous_transaction_count: int = Field(default=10, ge=0, description="Historical transaction count")
    previous_average_amount: float = Field(default=100.0, ge=0.0, description="Historical average transaction amount")
    failed_attempts: int = Field(default=0, ge=0, description="Number of recent failed attempts prior to this transaction")
    previous_device_known: bool = Field(default=True, description="Whether this device has been used by the user before")
    previous_merchant_known: bool = Field(default=True, description="Whether this merchant has been transacted with before")
    transactions_last_24h: int = Field(default=1, ge=0, description="Transaction count in the past 24 hours")
    is_location_consistent: bool = Field(default=True, description="Whether transaction location matches user's usual location")

    @field_validator("transaction_id", "user_id", "transaction_type", "merchant_id", "device_id", "location")
    @classmethod
    def validate_non_empty_strings(cls, value: str, info) -> str:
        """Ensure required string fields are not blank or empty strings."""
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty or blank")
        return value.strip()


class RuleResult(BaseModel):
    """Intermediate result produced by the deterministic Rule Engine."""
    rule_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score from rule evaluations")
    triggered_rules: List[str] = Field(default_factory=list, description="List of rule codes triggered")
    reasons: List[str] = Field(default_factory=list, description="Explanations corresponding to triggered rules")


class TransactionRiskOutput(BaseModel):
    """Final risk assessment response produced by the Fraud Risk Agent."""
    transaction_id: str = Field(..., description="Identifier matching the analyzed transaction")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated fraud probability from ML classifier [0-1]")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Estimated anomaly score from unsupervised model [0-1]")
    rule_score: float = Field(..., ge=0.0, le=1.0, description="Aggregate rule risk score [0-1]")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall fused risk score [0-1]")
    risk_level: RiskLevel = Field(..., description="Categorical risk classification: LOW, MEDIUM, HIGH, CRITICAL")
    risk_factors: List[str] = Field(default_factory=list, description="Explainable factual risk drivers")
    recommended_action: ActionType = Field(..., description="Recommended policy action")

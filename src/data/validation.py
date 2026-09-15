"""Validation helper functions for transaction data payloads."""
from typing import Any, Dict, List, Union
from pydantic import ValidationError

from src.schemas.transaction import TransactionInput


def validate_transaction_data(payload: Union[Dict[str, Any], TransactionInput]) -> TransactionInput:
    """Validate a raw dictionary or existing TransactionInput instance.

    Args:
        payload: Dictionary of transaction fields or TransactionInput instance.

    Returns:
        Validated TransactionInput model instance.

    Raises:
        ValueError: If validation fails.
    """
    if isinstance(payload, TransactionInput):
        return payload
    try:
        return TransactionInput(**payload)
    except ValidationError as err:
        error_msgs = [f"{e['loc'][0]}: {e['msg']}" for e in err.errors()]
        raise ValueError(f"Transaction validation error: {'; '.join(error_msgs)}") from err


def validate_transaction_batch(payloads: List[Dict[str, Any]]) -> List[TransactionInput]:
    """Validate a batch list of transaction dictionaries.

    Args:
        payloads: List of transaction dictionaries.

    Returns:
        List of validated TransactionInput models.
    """
    return [validate_transaction_data(item) for item in payloads]

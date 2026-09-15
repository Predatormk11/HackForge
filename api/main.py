"""FastAPI REST Service for Transaction Fraud & Risk Analysis."""
from contextlib import asynccontextmanager
import json
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.agent.orchestrator import AgentOrchestrator
from src.schemas.transaction import TransactionInput, TransactionRiskOutput

# Global orchestrator instance
orchestrator: AgentOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize the orchestrator and verify baseline models."""
    global orchestrator
    # Ensure models exist or are trained
    fraud_model_path = "models/fraud_model.pkl"
    anomaly_model_path = "models/anomaly_model.pkl"
    if not os.path.exists(fraud_model_path) or not os.path.exists(anomaly_model_path):
        from scripts.generate_demo_data import generate_synthetic_training_data, generate_demo_scenarios
        from scripts.train_fraud_model import train_fraud_model
        from scripts.train_anomaly_model import train_anomaly_model

        if not os.path.exists("data/raw/train_transactions.csv"):
            generate_synthetic_training_data()
            generate_demo_scenarios()
        if not os.path.exists(fraud_model_path):
            train_fraud_model()
        if not os.path.exists(anomaly_model_path):
            train_anomaly_model()

    orchestrator = AgentOrchestrator(config_path="config/config.yaml")
    yield


app = FastAPI(
    title="Fraud Detection & Transaction Risk Agent API",
    description="High-performance, explainable multi-signal financial transaction risk assessment engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin integration (Streamlit, UI frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        JSON status object {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post(
    "/api/v1/analyze",
    response_model=TransactionRiskOutput,
    status_code=status.HTTP_200_OK,
    tags=["Risk Analysis"],
    summary="Analyze transaction risk",
    description="Evaluates a transaction across Random Forest fraud classifier, Isolation Forest anomaly detector, and deterministic risk rules.",
)
def analyze_transaction(transaction: TransactionInput) -> TransactionRiskOutput:
    """Analyze a single transaction and produce explainable risk metrics and recommended actions."""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator(config_path="config/config.yaml")

    try:
        result = orchestrator.analyze_transaction(transaction)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating transaction risk: {str(exc)}",
        ) from exc


@app.get(
    "/api/v1/scenarios",
    tags=["Demo"],
    summary="Get predefined demo scenarios",
)
def get_demo_scenarios() -> Dict[str, Any]:
    """Return preconfigured demo scenarios for testing and demonstration."""
    sample_path = "data/sample/transactions.json"
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback inline scenarios if file not yet generated
    from scripts.generate_demo_data import generate_demo_scenarios
    return generate_demo_scenarios()

# 🛡️ Fraud Detection & Transaction Risk Agent

> **Phase-1 Working Prototype**: Multi-Signal AI & Deterministic Risk Orchestration Engine.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://pytest.org/)

---

## 📌 1. Project Overview & Problem Statement

Financial fraud causes billions of dollars in losses annually. Traditional rule-only systems are rigid and prone to high false-positive rates, while standalone black-box machine learning models lack auditability and cannot explain their recommendations.

The **Fraud Detection & Transaction Risk Agent** solves this by combining:
1. **Supervised Machine Learning (`RandomForestClassifier`)** trained on historical behavioral patterns.
2. **Unsupervised Anomaly Detection (`IsolationForest`)** to identify novel zero-day fraud outliers.
3. **Deterministic Expert Rule Engine** evaluating strict security heuristics (e.g., velocity spikes, nocturnal hours, device mismatches).
4. **Risk Fusion & Decision Engine** mapping multi-signal scores into actionable mitigation policies (`APPROVE`, `ADDITIONAL_VERIFICATION`, `HOLD_AND_REVIEW`, `BLOCK_AND_INVESTIGATE`).
5. **Evidence-Grounded Explanations** providing fully factual, non-hallucinated explanations for human analysts.

> ⚠️ **Important Disclaimer**: This system evaluates **estimated fraud risk** and outputs probabilistic risk signals; it is not a claim of definitive proof of fraud.

---

## 🏛️ 2. High-Level Architecture

```
Transaction Input
        ↓
Data & Feature Processor (14 deterministic features)
        ↓
Fraud Risk Agent / Orchestrator
        ↓
 ┌───────────────────────┬─────────────────────────┬──────────────────────┐
 │ Fraud Model (RF)      │ Anomaly Model (IF)      │ Rule Engine (Rules)  │
 │ [Fraud Prob: 0.0-1.0] │ [Anomaly Score: 0.0-1.0]│ [Rule Score: 0.0-1.0]│
 └───────────────────────┴─────────────────────────┴──────────────────────┘
        ↓
Risk Fusion Engine (0.60 * Fraud + 0.25 * Anomaly + 0.15 * Rule)
        ↓
Decision Engine (Thresholds: LOW < 0.30, MEDIUM < 0.70, HIGH < 0.90, CRITICAL >= 0.90)
        ↓
Explanation Generator (Factual evidence grounding)
        ↓
Final Output (TransactionRiskOutput)
```

For full technical specifications and sequence diagrams, see [docs/architecture.md](docs/architecture.md).

---

## 🚀 3. Features & Highlights

- **⚡ Zero External API Dependency**: Runs completely local out of the box without requiring paid LLM API keys or cloud infrastructure.
- **🎯 14-Feature Deterministic Pipeline**: Cleanly handles time deviations, historical ratios, velocity bursts, and device/merchant signals.
- **🔍 Explainable Risk Factors**: Replaces black-box uncertainty with human-readable, factual evidence lists.
- **🖥️ Dual Interface**: High-performance async **FastAPI** backend + Interactive **Streamlit** dashboard.
- **🧪 100% Test Coverage on Core Logic**: Unit tests for validation, feature engineering, rules, fusion formulas, decision tiers, and API endpoints.

---

## 📂 4. Repository Structure

```
fraud-risk-agent/
│
├── README.md                          # Project documentation & team guidelines
├── requirements.txt                   # Minimal project dependencies
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment configuration template
│
├── config/
│   └── config.yaml                    # Weights, risk thresholds & rule parameters
│
├── data/
│   ├── raw/                           # Generated synthetic training data (CSV)
│   ├── processed/                     # Preprocessed training matrices
│   └── sample/                        # Preconfigured benchmark demo scenarios (JSON)
│
├── models/
│   ├── .gitkeep                       # Serialized model persistence folder
│   ├── fraud_model.pkl                # Trained Random Forest classifier
│   └── anomaly_model.pkl              # Trained Isolation Forest detector
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Normalization & dataset loader
│   │   └── validation.py              # Pydantic validation helpers
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py     # 14-feature deterministic extractor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fraud_classifier.py        # Supervised Random Forest wrapper
│   │   └── anomaly_detector.py        # Unsupervised Isolation Forest wrapper
│   ├── rules/
│   │   ├── __init__.py
│   │   └── rule_engine.py             # 7 deterministic risk rules
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_fusion.py             # Weighted score fusion engine
│   │   └── decision_engine.py         # Policy mapping and tier classification
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # Central workflow orchestrator
│   │   └── explanation.py             # Evidence-grounded explanation builder
│   └── schemas/
│       ├── __init__.py
│       └── transaction.py             # Pydantic schemas (TransactionInput & Output)
│
├── api/
│   ├── __init__.py
│   └── main.py                        # FastAPI endpoints (/health, /api/v1/analyze)
│
├── app/
│   └── streamlit_app.py               # Interactive Streamlit dashboard
│
├── scripts/
│   ├── generate_demo_data.py          # Synthetic dataset & benchmark scenario generator
│   ├── train_fraud_model.py           # Supervised model training script
│   └── train_anomaly_model.py         # Unsupervised model training script
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py               # Validation & feature engineering tests
│   ├── test_rules.py                  # Rule engine unit tests
│   ├── test_risk.py                   # Risk fusion & decision engine tests
│   └── test_agent.py                  # Orchestrator & API integration tests
│
└── docs/
    └── architecture.md                # System design & Mermaid diagrams
```

---

## 🛠️ 5. Installation & Setup

### 5.1 Prerequisites
- Python 3.11 or higher
- Git

### 5.2 Clone and Setup Environment
```bash
# 1. Clone repository
git clone <repo-url>
cd fraud-risk-agent

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🧠 6. Training & Baseline Generation

Run the generation and training scripts to build the synthetic dataset and train the baseline models:

```bash
# Generate synthetic dataset and sample benchmark scenarios
python -m scripts.generate_demo_data

# Train the Random Forest Fraud Classifier (saved to models/fraud_model.pkl)
python -m scripts.train_fraud_model

# Train the Isolation Forest Anomaly Detector (saved to models/anomaly_model.pkl)
python -m scripts.train_anomaly_model
```

---

## 🚦 7. Running the Application

### 7.1 Start the FastAPI REST Backend
```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Docs (Swagger): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 7.2 Start the Streamlit Dashboard
In a separate terminal:
```bash
streamlit run app/streamlit_app.py
```
- Open browser at: `http://localhost:8501`

---

## 🧪 8. Running Automated Tests

Run the full pytest suite:
```bash
python -m pytest -v
```

---

## 📡 9. Sample API Request & Response

### Request (`POST /api/v1/analyze`)
```json
{
  "transaction_id": "TX_DEMO_002",
  "user_id": "USR_BOB_202",
  "amount": 850.00,
  "transaction_type": "PURCHASE",
  "timestamp": "2026-09-15T03:15:00",
  "merchant_id": "MERCH_LUXURY_999",
  "device_id": "DEV_UNRECOGNIZED_X1",
  "location": "Chicago, US",
  "account_age_days": 180,
  "previous_transaction_count": 40,
  "previous_average_amount": 95.00,
  "failed_attempts": 1,
  "previous_device_known": false,
  "previous_merchant_known": false,
  "transactions_last_24h": 3,
  "is_location_consistent": true
}
```

### Response
```json
{
  "transaction_id": "TX_DEMO_002",
  "fraud_probability": 0.6850,
  "anomaly_score": 0.5420,
  "rule_score": 0.5000,
  "risk_score": 0.6215,
  "risk_level": "MEDIUM",
  "risk_factors": [
    "Transaction amount ($850.00) is significantly above historical average ($95.00).",
    "New device detected.",
    "Transaction occurred during an unusual time (03:00).",
    "New merchant detected.",
    "Supervised fraud model detected high-risk behavioral pattern (estimated probability: 68.5%).",
    "Unsupervised anomaly detector flagged minor deviation from baseline norms (anomaly score: 54.2%)."
  ],
  "recommended_action": "ADDITIONAL_VERIFICATION"
}
```

---

## 👥 10. Team Module Ownership (4-Member Division)

| Team Member | Feature Branch | Assigned Modules & Focus Area |
| :--- | :--- | :--- |
| **Member 1 (Data Engineer)** | `feature/data-pipeline` | `src/data/`, `src/features/`, `src/schemas/`, historical rolling features, ingestion connectors. |
| **Member 2 (ML Engineer)** | `feature/fraud-model` | `src/models/`, `scripts/train_*`, model hyperparameter tuning, calibration, ROC-AUC validation. |
| **Member 3 (Risk & Agent Architect)** | `feature/risk-agent` | `src/rules/`, `src/risk/`, `src/agent/`, `api/`, weights configuration, policy action matrix. |
| **Member 4 (Full-Stack / UI Engineer)** | `feature/dashboard` | `app/streamlit_app.py`, visual metrics, batch CSV upload mode, analyst decision review flow. |

---

## 🤝 11. Git & Collaboration Workflow

To ensure stability across the 4-member team, **never commit directly to `main`**.

### Standard 10-Step Contribution Workflow:
1. **Pull latest `main`**:
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create feature branch**:
   ```bash
   git checkout -b feature/<feature-name>
   ```
3. **Implement changes** cleanly following typing and docstring standards.
4. **Run tests**:
   ```bash
   python -m pytest
   ```
5. **Commit using Conventional Commits**:
   - `feat:` (new feature)
   - `fix:` (bug fix)
   - `refactor:` (code refactoring)
   - `test:` (adding or updating tests)
   - `docs:` (documentation updates)
   - `chore:` (configuration or maintenance)
   ```bash
   git add .
   git commit -m "feat: add rolling velocity feature extraction"
   ```
6. **Push branch to remote**:
   ```bash
   git push origin feature/<feature-name>
   ```
7. **Open Pull Request (PR)** on GitHub targeting `main`.
8. **Code Review**: At least one peer review required.
9. **Merge PR** to `main`.
10. **Pull updated `main`** before starting next task.

---

## ⚖️ 12. Limitations & Future Roadmap

### Current Phase-1 Limitations
- Synthetic training data baseline used for initial cold start.
- Single-transaction synchronous evaluation (no async message broker like Kafka).
- Feature calculations rely on payload parameters rather than an external persistent feature store.

### Phase-2 Roadmap
- [ ] Connect with production database / Redis for automated historical state retrieval.
- [ ] Incorporate Graph Neural Network (GNN) / network clustering for money mule rings.
- [ ] Real-time feedback loop with analyst confirmed chargebacks.
- [ ] Containerization with Docker for production staging.

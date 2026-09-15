"""Streamlit Interactive Dashboard for Fraud Detection & Transaction Risk Agent."""
from datetime import datetime
import json
import os
import streamlit as st

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Fraud Risk Agent | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished look
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e222b;
        border: 1px solid #2d3342;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #90a4ae;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .badge-low {
        background-color: #1b4332;
        color: #74c69d;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: #5c4308;
        color: #ffd166;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #592512;
        color: #f77f00;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-critical {
        background-color: #4a0e17;
        color: #ef476f;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-factor-item {
        background: #1a202c;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    .action-box {
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-top: 10px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lazy import agent orchestrator
from src.agent.orchestrator import AgentOrchestrator
from src.schemas.transaction import TransactionInput, RiskLevel, ActionType


@st.cache_resource
def get_orchestrator():
    """Cache orchestrator instance to prevent reloading models on every interaction."""
    # Ensure baseline models exist
    if not os.path.exists("models/fraud_model.pkl") or not os.path.exists("models/anomaly_model.pkl"):
        from scripts.generate_demo_data import generate_synthetic_training_data, generate_demo_scenarios
        from scripts.train_fraud_model import train_fraud_model
        from scripts.train_anomaly_model import train_anomaly_model

        if not os.path.exists("data/raw/train_transactions.csv"):
            generate_synthetic_training_data()
            generate_demo_scenarios()
        train_fraud_model()
        train_anomaly_model()

    return AgentOrchestrator(config_path="config/config.yaml")


orchestrator = get_orchestrator()

# Header Section
st.title("🛡️ Fraud Detection & Transaction Risk Agent")
st.caption("Phase-1 Prototype | Multi-Signal AI & Deterministic Risk Orchestration Engine")

st.markdown("---")

# Demo Scenario Presets
sample_scenarios_file = "data/sample/transactions.json"
scenarios = {}
if os.path.exists(sample_scenarios_file):
    with open(sample_scenarios_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
else:
    from scripts.generate_demo_data import generate_demo_scenarios
    scenarios = generate_demo_scenarios()

# Sidebar: Quick Load Demo Scenarios
st.sidebar.header("🎯 Preloaded Demo Scenarios")
st.sidebar.markdown("Load benchmark transactions to evaluate system behavior:")

selected_scenario_key = st.sidebar.radio(
    "Choose Scenario:",
    options=list(scenarios.keys()),
    format_func=lambda k: scenarios[k]["name"],
)

active_scenario = scenarios[selected_scenario_key]
st.sidebar.info(f"**Description:**\n{active_scenario['description']}")
st.sidebar.markdown(f"**Expected Tier:** `{active_scenario['expected_level']}` | **Action:** `{active_scenario['expected_action']}`")

preset_data = active_scenario["data"]

# Session State for Input Form
if "loaded_scenario" not in st.session_state or st.session_state.loaded_scenario != selected_scenario_key:
    st.session_state.loaded_scenario = selected_scenario_key
    st.session_state.tx_id = preset_data.get("transaction_id", "TX_1001")
    st.session_state.user_id = preset_data.get("user_id", "USR_1001")
    st.session_state.amount = float(preset_data.get("amount", 100.0))
    st.session_state.tx_type = preset_data.get("transaction_type", "PURCHASE")
    st.session_state.merchant_id = preset_data.get("merchant_id", "MERCH_001")
    st.session_state.device_id = preset_data.get("device_id", "DEV_001")
    st.session_state.location = preset_data.get("location", "New York, US")
    st.session_state.account_age = int(preset_data.get("account_age_days", 30))
    st.session_state.prev_tx_count = int(preset_data.get("previous_transaction_count", 10))
    st.session_state.prev_avg_amt = float(preset_data.get("previous_average_amount", 100.0))
    st.session_state.failed_attempts = int(preset_data.get("failed_attempts", 0))
    st.session_state.prev_dev_known = bool(preset_data.get("previous_device_known", True))
    st.session_state.prev_merch_known = bool(preset_data.get("previous_merchant_known", True))
    st.session_state.tx_24h = int(preset_data.get("transactions_last_24h", 1))
    st.session_state.loc_match = bool(preset_data.get("is_location_consistent", True))

# Layout: Two Columns (Form on Left, Analysis Results on Right)
col_form, col_results = st.columns([1.1, 1.3], gap="large")

with col_form:
    st.subheader("📝 Transaction Details")

    with st.form("transaction_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tx_id = st.text_input("Transaction ID", value=st.session_state.tx_id)
            user_id = st.text_input("User ID", value=st.session_state.user_id)
            amount = st.number_input("Amount ($)", min_value=0.01, value=st.session_state.amount, step=10.0, format="%.2f")
            tx_type = st.selectbox(
                "Transaction Type",
                options=["PURCHASE", "TRANSFER", "WITHDRAWAL", "PAYMENT"],
                index=["PURCHASE", "TRANSFER", "WITHDRAWAL", "PAYMENT"].index(st.session_state.tx_type)
                if st.session_state.tx_type in ["PURCHASE", "TRANSFER", "WITHDRAWAL", "PAYMENT"]
                else 0,
            )

        with col_f2:
            merchant_id = st.text_input("Merchant ID", value=st.session_state.merchant_id)
            device_id = st.text_input("Device ID", value=st.session_state.device_id)
            location = st.text_input("Location", value=st.session_state.location)
            tx_timestamp_str = st.text_input("Timestamp (ISO 8601)", value=preset_data.get("timestamp", datetime.now().isoformat()))

        with st.expander("⚙️ Historical & Behavioral Context", expanded=True):
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                account_age_days = st.number_input("Account Age (Days)", min_value=0, value=st.session_state.account_age)
                previous_avg_amount = st.number_input("Historical Avg Amount ($)", min_value=0.0, value=st.session_state.prev_avg_amt, step=10.0, format="%.2f")
                failed_attempts = st.number_input("Recent Failed Attempts", min_value=0, max_value=20, value=st.session_state.failed_attempts)
                previous_device_known = st.checkbox("Device Recognized Before", value=st.session_state.prev_dev_known)

            with col_h2:
                previous_tx_count = st.number_input("Historical Tx Count", min_value=0, value=st.session_state.prev_tx_count)
                transactions_last_24h = st.number_input("Transactions in Last 24h", min_value=1, value=st.session_state.tx_24h)
                previous_merchant_known = st.checkbox("Merchant Recognized Before", value=st.session_state.prev_merch_known)
                is_location_consistent = st.checkbox("Location Matches User Profile", value=st.session_state.loc_match)

        submitted = st.form_submit_button("⚡ ANALYZE TRANSACTION", use_container_width=True, type="primary")

with col_results:
    st.subheader("📊 Risk Assessment Result")

    # Construct input model
    try:
        ts = datetime.fromisoformat(tx_timestamp_str)
    except Exception:
        ts = datetime.now()

    tx_input = TransactionInput(
        transaction_id=tx_id,
        user_id=user_id,
        amount=float(amount),
        transaction_type=tx_type,
        timestamp=ts,
        merchant_id=merchant_id,
        device_id=device_id,
        location=location,
        account_age_days=account_age_days,
        previous_transaction_count=previous_tx_count,
        previous_average_amount=float(previous_avg_amount),
        failed_attempts=failed_attempts,
        previous_device_known=previous_device_known,
        previous_merchant_known=previous_merchant_known,
        transactions_last_24h=transactions_last_24h,
        is_location_consistent=is_location_consistent,
    )

    # Perform analysis
    result = orchestrator.analyze_transaction(tx_input)

    # Color definitions based on risk level
    level_colors = {
        RiskLevel.LOW: ("#10b981", "#064e3b", "badge-low"),
        RiskLevel.MEDIUM: ("#f59e0b", "#78350f", "badge-medium"),
        RiskLevel.HIGH: ("#f97316", "#7c2d12", "badge-high"),
        RiskLevel.CRITICAL: ("#ef4444", "#7f1d1d", "badge-critical"),
    }

    color_fg, color_bg, badge_class = level_colors.get(
        result.risk_level,
        ("#94a3b8", "#1e293b", "badge-low"),
    )

    # Top Hero Summary Card
    st.markdown(
        f"""
        <div style="background-color: {color_bg}; border: 2px solid {color_fg}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.85rem; text-transform: uppercase; color: #cbd5e1; font-weight: 600;">Overall Risk Assessment</span>
                    <h1 style="color: {color_fg}; margin: 2px 0 0 0; font-size: 2.2rem;">
                        {result.risk_level.value} RISK ({result.risk_score:.1%})
                    </h1>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">RECOMMENDED ACTION</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; background: {color_fg}; padding: 6px 14px; border-radius: 6px; margin-top: 4px;">
                        {result.recommended_action.value}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breakdown Metrics Cards in 4 columns
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Fraud Prob. (ML)</div>
                <div class="metric-value" style="color: #60a5fa;">{result.fraud_probability:.1%}</div>
                <div style="font-size: 0.75rem; color: #64748b;">RandomForest (60% wt)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Anomaly Score</div>
                <div class="metric-value" style="color: #a78bfa;">{result.anomaly_score:.1%}</div>
                <div style="font-size: 0.75rem; color: #64748b;">IsolationForest (25% wt)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Rule Score</div>
                <div class="metric-value" style="color: #fb923c;">{result.rule_score:.1%}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Rule Engine (15% wt)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Combined Risk</div>
                <div class="metric-value" style="color: {color_fg};">{result.risk_score:.1%}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Fused Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Progress bar visualizer for scores
    st.markdown("**Risk Score Composition:**")
    st.progress(float(result.risk_score))

    # Explainable Risk Factors Section
    st.markdown("### 🔍 Explainable Risk Factors")
    if result.risk_factors:
        for factor in result.risk_factors:
            st.markdown(f'<div class="risk-factor-item">• {factor}</div>', unsafe_allow_html=True)
    else:
        st.info("No elevated risk anomalies detected. Standard baseline parameters.")

    # Technical Details Tabs
    with st.expander("🛠️ Analytical Breakdown & Raw Payload", expanded=False):
        tab1, tab2 = st.tabs(["JSON Output", "Extracted Feature Vector"])
        with tab1:
            st.json(result.model_dump())
        with tab2:
            feat_dict = orchestrator.feature_engineer.extract_features_dict(tx_input)
            st.dataframe(feat_dict, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This is a fraud-RISK evaluation prototype for transaction intelligence. Scores represent estimated statistical and behavioral risk signals, not conclusive proof of fraud."
)

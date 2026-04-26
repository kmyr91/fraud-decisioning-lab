# Objective:
# Streamlit dashboard for fraud decisioning results,
# policy comparison, champion-challenger evaluation, and monitoring.

from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Fraud Decisioning Dashboard", layout="wide")

project_root = Path(__file__).resolve().parent.parent

decision_results_path = project_root / "outputs" / "decision_results.csv"
policy_cost_path = project_root / "outputs" / "policy_cost_evaluation.csv"
champion_challenger_path = project_root / "outputs" / "champion_challenger_comparison.csv"
monitoring_summary_path = project_root / "outputs" / "monitoring_summary.csv"

decision_df = pd.read_csv(decision_results_path)
policy_df = pd.read_csv(policy_cost_path)
champion_df = pd.read_csv(champion_challenger_path)
monitoring_df = pd.read_csv(monitoring_summary_path)

st.title("Fraud Decisioning Dashboard")
st.markdown(
    "Production-style fraud decisioning prototype with policy backtesting, "
    "model comparison, and monitoring."
)

# Section 1: Overview
st.header("1. Decisioning Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total scored transactions", f"{len(decision_df):,}")
col2.metric("Average fraud score", f"{decision_df['fraud_score'].mean():.4f}")
col3.metric("Approve Rate", f"{(decision_df['decision'] == 'approve').mean() * 100:.2f}%")

decision_counts = (
    decision_df["decision"]
    .value_counts()
    .rename_axis("decision")
    .reset_index(name="count")
)

fraud_rate_by_decision = (
    decision_df.groupby("decision")["isFraud"]
    .mean()
    .reset_index()
    .rename(columns={"isFraud": "fraud_rate"})
)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Decision Counts")
    st.bar_chart(decision_counts.set_index("decision"))

with right_col:
    st.subheader("Fraud Rate by Decision")
    st.bar_chart(fraud_rate_by_decision.set_index("decision"))

# Section 2: Policy Comparison
st.header("2. Policy Comparison")
st.markdown("Tradeoff between fraud capture, review workload, and customer friction across policies.")
st.dataframe(policy_df, use_container_width=True)

policy_view = policy_df[
    [
        "policy_name",
        "approve_rate",
        "review_rate",
        "decline_rate",
        "fraud_capture_decline",
        "fraud_capture_review_decline",
        "false_decline_rate_good",
        "estimated_total_cost",
    ]
].copy()

st.subheader("Estimated Total Cost by Policy")
st.bar_chart(policy_view.set_index("policy_name")[["estimated_total_cost"]])

# Section 3: Champion vs Challenger
st.header("3. Champion vs Challenger")
st.markdown("Comparison of the Random Forest champion and XGBoost challenger under multiple operating policies.")
st.dataframe(champion_df, use_container_width=True)

best_config = champion_df.sort_values("estimated_total_cost").iloc[0]
st.success(
    f"Best configuration: {best_config['model_name']} with {best_config['policy_name']} policy "
    f"(estimated cost = {best_config['estimated_total_cost']})"
)

# Section 4: Monitoring Summary
st.header("4. Monitoring Summary")
st.markdown("Operational monitoring view of decision mix and scored transaction outcomes.")
st.dataframe(monitoring_df, use_container_width=True)

st.subheader("Decision Mix")
st.bar_chart(monitoring_df.set_index("decision")[["count"]])
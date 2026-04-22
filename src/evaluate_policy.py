# Objective:
# Evaluate business tradeoffs across multiple fraud decision policies
# using simple cost assumptions.

from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "outputs" / "all_policy_results.csv"

    df = pd.read_csv(file_path)

    # Simple business cost assumptions
    fraud_approved_cost = 100
    review_cost = 5
    good_declined_cost = 20

    summary_rows = []

    for policy_name in df["policy_name"].unique():
        policy_df = df[df["policy_name"] == policy_name].copy()

        total_transactions = len(policy_df)
        total_fraud = policy_df["isFraud"].sum()
        total_nonfraud = total_transactions - total_fraud

        approve_df = policy_df[policy_df["decision"] == "approve"]
        review_df = policy_df[policy_df["decision"] == "review"]
        decline_df = policy_df[policy_df["decision"] == "decline"]

        fraud_in_approve = approve_df["isFraud"].sum()
        fraud_in_review = review_df["isFraud"].sum()
        fraud_in_decline = decline_df["isFraud"].sum()

        good_in_approve = len(approve_df) - fraud_in_approve
        good_in_review = len(review_df) - fraud_in_review
        good_in_decline = len(decline_df) - fraud_in_decline

        total_review_count = len(review_df)

        estimated_cost = (
            fraud_in_approve * fraud_approved_cost
            + total_review_count * review_cost
            + good_in_decline * good_declined_cost
        )

        summary_rows.append({
            "policy_name": policy_name,
            "total_transactions": total_transactions,
            "total_fraud": int(total_fraud),
            "total_nonfraud": int(total_nonfraud),
            "approve_rate": len(approve_df) / total_transactions,
            "review_rate": len(review_df) / total_transactions,
            "decline_rate": len(decline_df) / total_transactions,
            "fraud_approved": int(fraud_in_approve),
            "fraud_reviewed": int(fraud_in_review),
            "fraud_declined": int(fraud_in_decline),
            "good_approved": int(good_in_approve),
            "good_reviewed": int(good_in_review),
            "good_declined": int(good_in_decline),
            "fraud_capture_decline": fraud_in_decline / total_fraud,
            "fraud_capture_review_decline": (fraud_in_review + fraud_in_decline) / total_fraud,
            "false_decline_rate_good": good_in_decline / total_nonfraud,
            "estimated_total_cost": estimated_cost
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("estimated_total_cost")

    print("=" * 80)
    print("POLICY COST EVALUATION")
    print("=" * 80)
    print(summary_df)

    print("\nBest policy by estimated total cost:")
    print(summary_df.iloc[0][["policy_name", "estimated_total_cost"]])

    save_path = project_root / "outputs" / "policy_cost_evaluation.csv"
    summary_df.to_csv(save_path, index=False)
    print(f"\nSaved policy cost evaluation to: {save_path}")


if __name__ == "__main__":
    main()
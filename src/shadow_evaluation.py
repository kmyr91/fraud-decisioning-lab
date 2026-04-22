# Objective:
# Simulate a shadow deployment decision between a champion and challenger model
# based on backtested policy and business-cost results.

from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "outputs" / "champion_challenger_comparison.csv"

    df = pd.read_csv(file_path)

    champion_df = df[df["model_name"] == "random_forest_champion"].copy()
    challenger_df = df[df["model_name"] == "xgboost_challenger"].copy()

    champion_best = champion_df.sort_values("estimated_total_cost").iloc[0]
    challenger_best = challenger_df.sort_values("estimated_total_cost").iloc[0]

    print("=" * 80)
    print("SHADOW DEPLOYMENT EVALUATION")
    print("=" * 80)

    print("\nChampion best configuration:")
    print(champion_best[[
        "model_name",
        "policy_name",
        "roc_auc",
        "fraud_capture_decline",
        "fraud_capture_review_decline",
        "false_decline_rate_good",
        "estimated_total_cost"
    ]])

    print("\nChallenger best configuration:")
    print(challenger_best[[
        "model_name",
        "policy_name",
        "roc_auc",
        "fraud_capture_decline",
        "fraud_capture_review_decline",
        "false_decline_rate_good",
        "estimated_total_cost"
    ]])

    promotion_conditions = [
        challenger_best["estimated_total_cost"] < champion_best["estimated_total_cost"],
        challenger_best["fraud_capture_review_decline"] >= champion_best["fraud_capture_review_decline"] - 0.02,
        challenger_best["false_decline_rate_good"] <= champion_best["false_decline_rate_good"] + 0.002
    ]

    should_promote = all(promotion_conditions)

    print("\nPromotion criteria:")
    print("1. Lower estimated total cost than champion")
    print("2. Fraud capture in review+decline not materially worse")
    print("3. False decline rate not materially worse")

    print("\nCriteria results:")
    print(f"Lower cost: {promotion_conditions[0]}")
    print(f"Comparable fraud capture: {promotion_conditions[1]}")
    print(f"Comparable false decline rate: {promotion_conditions[2]}")

    if should_promote:
        decision = "PROMOTE CHALLENGER"
        rationale = "Challenger meets all promotion criteria and can replace the champion."
    else:
        decision = "KEEP CHAMPION / SHADOW ONLY"
        rationale = "Challenger does not beat the champion on the defined policy criteria."

    print("\nFinal decision:")
    print(decision)
    print(rationale)

    summary = pd.DataFrame([
        {
            "champion_model": champion_best["model_name"],
            "champion_policy": champion_best["policy_name"],
            "champion_cost": champion_best["estimated_total_cost"],
            "challenger_model": challenger_best["model_name"],
            "challenger_policy": challenger_best["policy_name"],
            "challenger_cost": challenger_best["estimated_total_cost"],
            "promote_challenger": should_promote,
            "final_decision": decision,
            "rationale": rationale
        }
    ])

    save_path = project_root / "outputs" / "shadow_evaluation_summary.csv"
    summary.to_csv(save_path, index=False)

    print(f"\nSaved shadow evaluation summary to: {save_path}")


if __name__ == "__main__":
    main()
# Objective:
# Train the baseline model, generate fraud scores,
# and compare multiple approve / review / decline policies.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def assign_decision(score, approve_threshold, review_threshold):
    if score < approve_threshold:
        return "approve"
    elif score < review_threshold:
        return "review"
    else:
        return "decline"


def evaluate_policy(results_df, policy_name):
    total_transactions = len(results_df)
    total_fraud = results_df["isFraud"].sum()
    total_nonfraud = total_transactions - total_fraud

    approve_df = results_df[results_df["decision"] == "approve"]
    review_df = results_df[results_df["decision"] == "review"]
    decline_df = results_df[results_df["decision"] == "decline"]

    fraud_in_approve = approve_df["isFraud"].sum()
    fraud_in_review = review_df["isFraud"].sum()
    fraud_in_decline = decline_df["isFraud"].sum()

    good_in_approve = len(approve_df) - fraud_in_approve
    good_in_review = len(review_df) - fraud_in_review
    good_in_decline = len(decline_df) - fraud_in_decline

    metrics = {
        "policy_name": policy_name,
        "approve_rate": len(approve_df) / total_transactions,
        "review_rate": len(review_df) / total_transactions,
        "decline_rate": len(decline_df) / total_transactions,
        "fraud_capture_decline": fraud_in_decline / total_fraud,
        "fraud_capture_review_decline": (fraud_in_review + fraud_in_decline) / total_fraud,
        "false_decline_rate_good": good_in_decline / total_nonfraud,
        "fraud_rate_approve": approve_df["isFraud"].mean() if len(approve_df) > 0 else 0,
        "fraud_rate_review": review_df["isFraud"].mean() if len(review_df) > 0 else 0,
        "fraud_rate_decline": decline_df["isFraud"].mean() if len(decline_df) > 0 else 0,
        "fraud_approved": fraud_in_approve,
        "fraud_reviewed": fraud_in_review,
        "fraud_declined": fraud_in_decline,
        "good_approved": good_in_approve,
        "good_reviewed": good_in_review,
        "good_declined": good_in_decline,
    }

    return metrics


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "outputs" / "baseline_dataset.csv"

    df = pd.read_csv(data_path)

    target_col = "isFraud"
    drop_cols = ["TransactionID"]

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-999))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    fraud_scores = pipeline.predict_proba(X_test)[:, 1]

    base_results = pd.DataFrame({
        "isFraud": y_test.reset_index(drop=True),
        "fraud_score": fraud_scores
    })

    policies = [
        {"policy_name": "conservative", "approve_threshold": 0.05, "review_threshold": 0.40},
        {"policy_name": "balanced", "approve_threshold": 0.10, "review_threshold": 0.60},
        {"policy_name": "aggressive", "approve_threshold": 0.15, "review_threshold": 0.75},
    ]

    all_policy_metrics = []
    all_policy_results = []

    for policy in policies:
        results = base_results.copy()
        results["decision"] = results["fraud_score"].apply(
            lambda score: assign_decision(
                score,
                policy["approve_threshold"],
                policy["review_threshold"]
            )
        )
        results["policy_name"] = policy["policy_name"]

        metrics = evaluate_policy(results, policy["policy_name"])
        all_policy_metrics.append(metrics)
        all_policy_results.append(results)

        print("\n" + "=" * 70)
        print(f"POLICY: {policy['policy_name']}")
        print("=" * 70)
        print("Decision counts:")
        print(results["decision"].value_counts())

        print("\nFraud rate by decision:")
        print(results.groupby("decision")["isFraud"].mean())

        print("\nCounts by decision and fraud label:")
        print(pd.crosstab(results["decision"], results["isFraud"]))

    metrics_df = pd.DataFrame(all_policy_metrics)
    results_df = pd.concat(all_policy_results, ignore_index=True)

    metrics_save_path = project_root / "outputs" / "policy_comparison.csv"
    results_save_path = project_root / "outputs" / "all_policy_results.csv"

    metrics_df.to_csv(metrics_save_path, index=False)
    results_df.to_csv(results_save_path, index=False)

    print("\n" + "=" * 70)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 70)
    print(
        metrics_df[
            [
                "policy_name",
                "approve_rate",
                "review_rate",
                "decline_rate",
                "fraud_capture_decline",
                "fraud_capture_review_decline",
                "false_decline_rate_good"
            ]
        ]
    )

    print(f"\nSaved policy comparison to: {metrics_save_path}")
    print(f"Saved all policy results to: {results_save_path}")


if __name__ == "__main__":
    main()
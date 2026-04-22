# Objective:
# Compare a champion model (Random Forest) and a challenger model (XGBoost)
# using the same fraud decision policies and estimated business cost.

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier


def assign_decision(score, approve_threshold, review_threshold):
    if score < approve_threshold:
        return "approve"
    elif score < review_threshold:
        return "review"
    else:
        return "decline"


def evaluate_policy(results_df, model_name, policy_name):
    total_transactions = len(results_df)
    total_fraud = results_df["isFraud"].sum()
    total_nonfraud = total_transactions - total_fraud

    approve_df = results_df[results_df["decision"] == "approve"]
    review_df = results_df[results_df["decision"] == "review"]
    decline_df = results_df[results_df["decision"] == "decline"]

    fraud_in_approve = int(approve_df["isFraud"].sum())
    fraud_in_review = int(review_df["isFraud"].sum())
    fraud_in_decline = int(decline_df["isFraud"].sum())

    good_in_approve = len(approve_df) - fraud_in_approve
    good_in_review = len(review_df) - fraud_in_review
    good_in_decline = len(decline_df) - fraud_in_decline

    # Illustrative business costs
    fraud_approved_cost = 100
    review_cost = 5
    good_declined_cost = 20

    estimated_total_cost = (
        fraud_in_approve * fraud_approved_cost
        + len(review_df) * review_cost
        + good_in_decline * good_declined_cost
    )

    metrics = {
        "model_name": model_name,
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
        "good_approved": int(good_in_approve),
        "good_reviewed": int(good_in_review),
        "good_declined": int(good_in_decline),
        "estimated_total_cost": estimated_total_cost,
    }

    return metrics


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

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

    return preprocessor


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "outputs" / "baseline_dataset.csv"

    df = pd.read_csv(data_path)

    target_col = "isFraud"
    drop_cols = ["TransactionID"]

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "random_forest_champion": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "xgboost_challenger": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    policies = [
        {"policy_name": "conservative", "approve_threshold": 0.05, "review_threshold": 0.40},
        {"policy_name": "balanced", "approve_threshold": 0.10, "review_threshold": 0.60},
        {"policy_name": "aggressive", "approve_threshold": 0.15, "review_threshold": 0.75},
    ]

    all_metrics = []

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL: {model_name}")
        print("=" * 80)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        fraud_scores = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, fraud_scores)

        print(f"ROC AUC: {auc:.6f}")

        base_results = pd.DataFrame({
            "isFraud": y_test.reset_index(drop=True),
            "fraud_score": fraud_scores
        })

        for policy in policies:
            results = base_results.copy()
            results["decision"] = results["fraud_score"].apply(
                lambda score: assign_decision(
                    score,
                    policy["approve_threshold"],
                    policy["review_threshold"]
                )
            )

            metrics = evaluate_policy(
                results,
                model_name=model_name,
                policy_name=policy["policy_name"]
            )
            metrics["roc_auc"] = auc
            all_metrics.append(metrics)

            print("\n" + "-" * 60)
            print(f"Policy: {policy['policy_name']}")
            print("-" * 60)
            print(results["decision"].value_counts())
            print("\nFraud rate by decision:")
            print(results.groupby("decision")["isFraud"].mean())

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values(
        by=["estimated_total_cost", "roc_auc"],
        ascending=[True, False]
    )

    save_path = project_root / "outputs" / "champion_challenger_comparison.csv"
    comparison_df.to_csv(save_path, index=False)

    print("\n" + "=" * 80)
    print("CHAMPION VS CHALLENGER SUMMARY")
    print("=" * 80)
    print(
        comparison_df[
            [
                "model_name",
                "policy_name",
                "roc_auc",
                "approve_rate",
                "review_rate",
                "decline_rate",
                "fraud_capture_decline",
                "fraud_capture_review_decline",
                "false_decline_rate_good",
                "estimated_total_cost",
            ]
        ]
    )

    print(f"\nSaved comparison to: {save_path}")


if __name__ == "__main__":
    main()
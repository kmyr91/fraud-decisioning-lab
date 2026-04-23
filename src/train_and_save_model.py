# Objective:
# Train the champion fraud model and save the full pipeline for later scoring.

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "outputs" / "baseline_dataset.csv"
    models_path = project_root / "models"
    models_path.mkdir(exist_ok=True)

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

    print("Training champion model...")
    pipeline.fit(X_train, y_train)

    fraud_scores = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, fraud_scores)

    print(f"Champion ROC AUC: {auc:.6f}")

    model_artifact = {
        "model_name": "random_forest_champion",
        "pipeline": pipeline,
        "approve_threshold": 0.10,
        "review_threshold": 0.60,
        "feature_columns": X.columns.tolist(),
        "roc_auc": auc
    }

    save_path = models_path / "random_forest_champion.joblib"
    joblib.dump(model_artifact, save_path)

    print(f"Saved model artifact to: {save_path}")


if __name__ == "__main__":
    main()
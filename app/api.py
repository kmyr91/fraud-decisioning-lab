# Objective:
# Expose the champion fraud model through a FastAPI scoring endpoint.

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "random_forest_champion.joblib"

model_artifact = joblib.load(model_path)
pipeline = model_artifact["pipeline"]
model_name = model_artifact["model_name"]
approve_threshold = model_artifact["approve_threshold"]
review_threshold = model_artifact["review_threshold"]
feature_columns = model_artifact["feature_columns"]


app = FastAPI(title="Fraud Decisioning API")


class TransactionRequest(BaseModel):
    TransactionDT: float
    TransactionAmt: float
    ProductCD: str
    card1: float
    card2: float
    card3: float
    card4: str
    card5: float
    card6: str
    addr1: float
    addr2: float
    dist1: float
    dist2: float
    P_emaildomain: str
    R_emaildomain: str
    DeviceType: str
    DeviceInfo: str
    id_30: str
    id_31: str
    id_32: str
    id_33: str


def assign_decision(score: float) -> str:
    if score < approve_threshold:
        return "approve"
    elif score < review_threshold:
        return "review"
    return "decline"


@app.get("/")
def root():
    return {
        "message": "Fraud Decisioning API is running",
        "model_name": model_name
    }


@app.post("/score")
def score_transaction(request: TransactionRequest):
    input_df = pd.DataFrame([request.dict()])

    input_df = input_df.reindex(columns=feature_columns)

    fraud_score = float(pipeline.predict_proba(input_df)[0][1])
    decision = assign_decision(fraud_score)

    return {
        "model_name": model_name,
        "fraud_score": round(fraud_score, 6),
        "decision": decision,
        "approve_threshold": approve_threshold,
        "review_threshold": review_threshold
    }
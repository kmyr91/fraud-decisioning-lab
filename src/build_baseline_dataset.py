# Objective:
# Build a smaller baseline dataset for the first fraud model.

from pathlib import Path
import pandas as pd

def main():
    data_path = Path(__file__).resolve().parent.parent / "data"
    output_path = Path(__file__).resolve().parent.parent / "outputs"

    transaction_path = data_path / 'train_transaction.csv'
    identity_path = data_path / 'train_identity.csv'

    transactions = pd.read_csv(transaction_path)
    identities = pd.read_csv(identity_path)

    df = transactions.merge(identities, on='TransactionID', how='left')

    selected_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_32",
        "id_33"
    ]
        
    baseline_df = df[selected_columns].copy()
    baseline_df = baseline_df.loc[:, ~baseline_df.columns.duplicated()]
    print("Baseline dataset shape before cleaning:", baseline_df.shape)

    categorical_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
                        "id_30","id_31","id_32","id_33"]
    numerical_cols = ["TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]

    for col in categorical_cols:
        baseline_df[col] = baseline_df[col].fillna("missing").astype(str)

    for col in numerical_cols:
        baseline_df[col] = baseline_df[col].fillna(-999)

    output_path.mkdir(exist_ok=True)
    save_path = output_path / "baseline_dataset.csv"
    baseline_df.to_csv(save_path, index=False)
    print("Baseline dataset shape after cleaning:", baseline_df.shape)
    print(f"Baseline dataset saved to: {save_path}")
    print("\nFraud rate:", baseline_df['isFraud'].mean())


if __name__ == "__main__":
    main()
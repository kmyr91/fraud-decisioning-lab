# Objective:
# Merge transaction and identity data on TransactionID
# and inspect the merged dataset.

from pathlib import Path
import pandas as pd


def main():
    data_path = Path(__file__).resolve().parent.parent / "data"

    transaction_path = data_path / "train_transaction.csv"
    identity_path = data_path / "train_identity.csv"

    transactions = pd.read_csv(transaction_path)
    identities = pd.read_csv(identity_path)

    merged_data = pd.merge(
        transactions,
        identities,
        on="TransactionID",
        how="left"
    )

    print("Merged shape:", merged_data.shape)
    print("\nFirst 5 rows:")
    print(merged_data.head())


if __name__ == "__main__":
    main()
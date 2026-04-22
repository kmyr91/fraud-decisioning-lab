# Objective:
# Profile the merged fraud dataset:
# target balance, missingness, and a few basic feature summaries.

from pathlib import Path
import pandas as pd

def main():
    data_path = Path(__file__).resolve().parent.parent / "data"
    transcation_path = data_path / 'train_transaction.csv'
    identity_path = data_path / 'train_identity.csv'

    transactions = pd.read_csv(transcation_path)
    identities = pd.read_csv(identity_path)


    df = transactions.merge(identities, on='TransactionID', how='left')
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
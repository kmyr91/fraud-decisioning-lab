# Objective:
# Load the fraud dataset files from the data folder
# and inspect their structure.

from pathlib import Path
import pandas as pd


def main():
    data_path = Path(__file__).resolve().parent.parent / "data"

    file_names = [
        "train_transaction.csv",
        "train_identity.csv"
    ]

    for file_name in file_names:
        file_path = data_path / file_name

        try:
            df = pd.read_csv(file_path)
            print("=" * 70)
            print(f"FILE: {file_name}")
            print("=" * 70)
            print("Shape:", df.shape)
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head())
            print("\n")
        except Exception as e:
            print(f"Could not read {file_name}: {e}")


if __name__ == "__main__":
    main()
# Objective:
# Monitor key decisioning outputs such as score distribution,
# decision mix, and fraud rate by decision bucket.

from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "outputs" / "decision_results.csv"

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)

    print("\nTotal scored transactions:")
    print(len(df))

    print("\nAverage fraud score:")
    print(round(df["fraud_score"].mean(), 6))

    print("\nFraud score summary:")
    print(df["fraud_score"].describe())

    print("\nDecision mix:")
    decision_mix = df["decision"].value_counts(normalize=True).sort_index()
    print(decision_mix)

    print("\nDecision counts:")
    print(df["decision"].value_counts())

    print("\nFraud rate by decision:")
    print(df.groupby("decision")["isFraud"].mean())

    print("\nFraud count by decision:")
    print(df.groupby("decision")["isFraud"].sum())

    print("\nGood transaction count by decision:")
    good_counts = df.groupby("decision").apply(lambda x: (x["isFraud"] == 0).sum())
    print(good_counts)

    save_path = project_root / "outputs" / "monitoring_summary.csv"

    summary_df = pd.DataFrame({
        "decision": df["decision"].value_counts().index,
        "count": df["decision"].value_counts().values
    })

    summary_df.to_csv(save_path, index=False)
    print(f"\nSaved monitoring summary to: {save_path}")


if __name__ == "__main__":
    main()
"""
Data validation checks for the Telco Customer Churn dataset.
Run standalone or import validate() into a pipeline.
"""
import pandas as pd


def validate(df: pd.DataFrame) -> dict:
    """Run validation checks and return a report dict. Raises AssertionError on critical failures."""
    report = {}

    # Critical: no duplicate customer IDs
    dup_count = df['customerID'].duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate customerID(s)"
    report['duplicate_ids'] = dup_count

    # Critical: TotalCharges non-numeric rows should only occur at tenure == 0
    non_numeric_mask = pd.to_numeric(df['TotalCharges'], errors='coerce').isna()
    bad_rows = df.loc[non_numeric_mask]
    assert (bad_rows['tenure'] == 0).all(), \
        "Found non-numeric TotalCharges NOT explained by tenure == 0 — investigate before filling"
    report['total_charges_blank_rows'] = int(non_numeric_mask.sum())

    # Target balance (informational, not an assertion)
    report['churn_rate'] = df['Churn'].value_counts(normalize=True).to_dict()

    # No unexpected nulls anywhere
    null_counts = df.isnull().sum()
    report['null_counts'] = null_counts[null_counts > 0].to_dict()

    return report


if __name__ == "__main__":
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    result = validate(df)
    print(result)
    print(" Validation passed")
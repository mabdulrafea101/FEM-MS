import pandas as pd
from pipeline.config import DATA_PATH, TARGET_COLS


def load_dataset(path=DATA_PATH):
    """Load the frozen ANSYS dataset and derive the family column."""
    df = pd.read_excel(path, sheet_name="ML Dataset")
    df["family"] = df["combination_code"]
    return df


def run_qc(df):
    """Run the dataset audit from Ch. 3.9; raise on any failed check."""
    checks = {
        "total_cases": int(len(df)),
        "unique_case_ids": int(df["case_id"].nunique()),
        "family_balance": df["family"].value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "qc_all_pass": bool((df["preanalysis_qc"] == "PASS").all()),
        "targets_positive": bool(
            all(c in df.columns for c in TARGET_COLS)
            and (df[TARGET_COLS] > 0).all().all()
        ),
    }
    assert checks["total_cases"] == 1000, "expected 1000 cases"
    assert checks["unique_case_ids"] == 1000, "expected unique case ids"
    assert checks["missing_values"] == 0, "expected no missing values"
    assert checks["qc_all_pass"], "expected all QC PASS"
    assert set(checks["family_balance"]) == {"FF", "SS"}, "expected FF/SS families"
    assert checks["targets_positive"], "expected positive target frequencies"
    return checks

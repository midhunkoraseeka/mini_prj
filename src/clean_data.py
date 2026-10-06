"""Step 2: clean the raw data with pandas."""
import numpy as np
import pandas as pd

from config import CLEAN_CSV, FEATURES, RANGES, RAW_CSV, TARGET


def clean(df: pd.DataFrame):
    log = {"rows_raw": len(df)}
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Formatting: normalise the target label to 0/1
    t = df[TARGET].astype(str).str.strip().str.lower()
    df[TARGET] = t.map({"yes": 1, "y": 1, "placed": 1, "1": 1, "no": 0, "n": 0, "not placed": 0, "0": 0})
    log["bad_target_rows_dropped"] = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET])

    # Duplicates
    before = len(df)
    df = df.drop_duplicates()
    if "Student_ID" in df:
        df = df.drop_duplicates(subset="Student_ID")
    log["duplicates_removed"] = before - len(df)

    # Types + out-of-range values -> NaN (then imputed)
    out_of_range = 0
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lo, hi = RANGES[c]
        bad = (df[c] < lo) | (df[c] > hi)
        out_of_range += int(bad.sum())
        df.loc[bad, c] = np.nan
    log["out_of_range_set_missing"] = out_of_range

    # Missing values -> median imputation
    log["missing_imputed"] = int(df[FEATURES].isna().sum().sum())
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    for c in ["Internships", "Projects", "Backlogs", "Certifications"]:
        df[c] = df[c].round().astype(int)

    df[TARGET] = df[TARGET].astype(int)
    log["rows_clean"] = len(df)
    return df.reset_index(drop=True), log


if __name__ == "__main__":
    df, log = clean(pd.read_csv(RAW_CSV))
    df.to_csv(CLEAN_CSV, index=False)
    print(log)

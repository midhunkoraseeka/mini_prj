import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import clean_data
import generate_data
import recommend
from config import FEATURES, RANGES, TARGET

STRONG = dict(CGPA=9.2, Attendance=95, Internships=3, Projects=5, Coding_Score=90,
              Aptitude_Score=85, Backlogs=0, Certifications=4)
WEAK = dict(CGPA=5.2, Attendance=55, Internships=0, Projects=0, Coding_Score=20,
            Aptitude_Score=30, Backlogs=5, Certifications=0)


def test_cleaning_removes_mess():
    df, log = clean_data.clean(generate_data.generate(300))
    assert df.duplicated().sum() == 0
    assert df[FEATURES].isna().sum().sum() == 0
    assert set(df[TARGET].unique()) <= {0, 1}
    for c, (lo, hi) in RANGES.items():
        assert df[c].between(lo, hi).all()
    assert log["rows_clean"] == len(df) < log["rows_raw"]


def test_model_ordering_and_plan():
    bundle = recommend.load_model()
    _, p_strong = recommend.predict(bundle, STRONG)
    _, p_weak = recommend.predict(bundle, WEAK)
    assert p_strong > 0.8 > 0.3 > p_weak
    plan = recommend.action_plan(bundle, WEAK)
    assert plan and all(p["gain"] >= 0 for p in plan)
    assert recommend.action_plan(bundle, STRONG) == []


def test_explain_and_percentiles():
    bundle = recommend.load_model()
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "student_placement_clean.csv")
    ex = recommend.explain(bundle, WEAK, df[FEATURES].median().to_dict())
    assert len(ex) == len(FEATURES)
    pct = recommend.percentiles(df, STRONG)
    assert all(0 <= v <= 100 for v in pct.values()) and pct["CGPA"] > 90

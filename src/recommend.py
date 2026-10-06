"""Steps 6-7: prediction, explanation and improvement suggestions."""
import joblib
import pandas as pd

from config import FEATURES, MODEL_PATH

# (feature, healthy target, higher_is_better, suggestion)
BENCHMARKS = [
    ("CGPA", 7.5, True, "Raise your CGPA: focus on core subjects and aim for 7.5+ (many companies filter at 7.0)."),
    ("Attendance", 80, True, "Improve attendance to 80%+; it reflects consistency and avoids eligibility issues."),
    ("Internships", 1, True, "Do at least one internship for practical exposure."),
    ("Projects", 3, True, "Build 3+ projects (ideally deployed, with GitHub links) to show applied skills."),
    ("Coding_Score", 65, True, "Practise DSA daily (LeetCode/HackerRank) and mock coding rounds to push your coding score above 65."),
    ("Aptitude_Score", 65, True, "Practise quantitative, logical and verbal aptitude sets regularly."),
    ("Backlogs", 0, False, "Clear your backlogs as soon as possible; many companies reject candidates with active backlogs."),
    ("Certifications", 2, True, "Earn 2+ relevant certifications (cloud, data, programming) to strengthen your profile."),
]


def load_model():
    return joblib.load(MODEL_PATH)


def predict(bundle, student: dict):
    X = pd.DataFrame([student])[bundle["features"]]
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    return int(prob >= 0.5), prob


def predict_many(bundle, df: pd.DataFrame) -> pd.Series:
    return pd.Series(bundle["model"].predict_proba(df[bundle["features"]])[:, 1], index=df.index)


def explain(bundle, student: dict, reference: dict) -> pd.DataFrame:
    """Model-agnostic local explanation: how much does each feature move this student's probability
    compared with a typical student (reference = dataset medians)? Positive = helps, negative = hurts."""
    _, base = predict(bundle, student)
    rows = []
    for f in FEATURES:
        _, p = predict(bundle, {**student, f: reference[f]})
        rows.append({"feature": f, "value": student[f], "typical": reference[f], "impact": base - p})
    return pd.DataFrame(rows).sort_values("impact")


def action_plan(bundle, student: dict):
    """Simulate fixing each weak area on its own; rank by the probability gain it would bring."""
    _, base = predict(bundle, student)
    plan = []
    for feat, target, higher, tip in BENCHMARKS:
        v = student[feat]
        if (higher and v >= target) or (not higher and v <= target):
            continue
        _, p = predict(bundle, {**student, feat: target})
        plan.append({"feature": feat, "value": v, "target": target, "gain": p - base,
                     "new_prob": p, "suggestion": tip})
    return sorted(plan, key=lambda d: -d["gain"])


def percentiles(df: pd.DataFrame, student: dict) -> dict:
    """Percentile rank of the student within the cohort for each feature (backlogs inverted: higher is better)."""
    out = {}
    for f in FEATURES:
        pct = (df[f] < student[f]).mean() * 100 + (df[f] == student[f]).mean() * 50
        out[f] = 100 - pct if f == "Backlogs" else pct
    return out

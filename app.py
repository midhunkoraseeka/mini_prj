"""Vercel/Flask entry point: serves the dashboard (public/index.html) and a small JSON API.

Local:   python app.py            -> http://localhost:5000
Vercel:  detects the top-level `app` object automatically.
(The Streamlit version of the dashboard lives in dashboard/streamlit_app.py.)
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import clean_data
import recommend
from config import CLEAN_CSV, FEATURES, METRICS_PATH, RANGES, TARGET

app = Flask(__name__, static_folder=None)

_df = pd.read_csv(CLEAN_CSV)
_bundle = recommend.load_model()
_metrics = json.loads(METRICS_PATH.read_text())
_median = _df[FEATURES].median().to_dict()
MAX_BATCH_ROWS = 5000


def _summary():
    placed = _df[TARGET] == 1
    prof = _df.groupby(TARGET)[FEATURES].mean()
    return {
        "features": FEATURES,
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "defaults": {"CGPA": 7.0, "Attendance": 80, "Internships": 0, "Projects": 2,
                     "Coding_Score": 55, "Aptitude_Score": 60, "Backlogs": 0, "Certifications": 1},
        "median": _median,
        "n": len(_df),
        "placement_rate": float(placed.mean()),
        "means": _df[FEATURES].mean().round(2).to_dict(),
        "profile": {"placed": prof.loc[1].round(2).to_dict(), "not_placed": prof.loc[0].round(2).to_dict()},
        "rows": {c: _df[c].tolist() for c in FEATURES + [TARGET]},
        "corr": {"labels": FEATURES + [TARGET],
                 "matrix": _df[FEATURES + [TARGET]].corr().round(3).values.tolist()},
        "metrics": _metrics,
    }


_SUMMARY = _summary()


def _parse_student(payload):
    student = {}
    for f in FEATURES:
        try:
            v = float(payload[f])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{f}' is missing or not a number")
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            raise ValueError(f"'{f}' must be between {lo} and {hi}")
        student[f] = int(round(v)) if isinstance(lo, int) else v
    return student


@app.get("/")
def index():
    return send_from_directory(ROOT / "public", "index.html")


@app.get("/api/summary")
def summary():
    return jsonify(_SUMMARY)


@app.post("/api/predict")
def predict():
    try:
        student = _parse_student(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    label, prob = recommend.predict(_bundle, student)
    out = {"label": label, "probability": prob}
    if request.args.get("simple"):  # what-if sliders only need the probability
        return jsonify(out)
    ex = recommend.explain(_bundle, student, _median)
    out["explain"] = ex.to_dict(orient="records")
    out["plan"] = recommend.action_plan(_bundle, student)
    out["percentiles"] = recommend.percentiles(_df, student)
    return jsonify(out)


@app.post("/api/batch")
def batch():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        raw = pd.read_csv(io.StringIO(payload.get("csv", "")))
    except Exception:
        return jsonify(error="Could not read the file as CSV"), 400
    raw.columns = raw.columns.str.strip()
    missing = [f for f in FEATURES if f not in raw.columns]
    if missing:
        return jsonify(error=f"Missing columns: {', '.join(missing)}"), 400
    if len(raw) > MAX_BATCH_ROWS:
        return jsonify(error=f"Too many rows (max {MAX_BATCH_ROWS})"), 400
    raw[TARGET] = 0  # placeholder so the shared cleaning code can be reused
    clean, _ = clean_data.clean(raw)
    scored = clean.drop(columns=TARGET)
    scored["Placement_Probability"] = recommend.predict_many(_bundle, scored).round(3)
    scored["Prediction"] = scored["Placement_Probability"].ge(0.5).map({True: "Placed", False: "Not Placed"})
    scored = scored.sort_values("Placement_Probability")
    return app.response_class(scored.to_json(orient="records"), mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=False, port=5000)

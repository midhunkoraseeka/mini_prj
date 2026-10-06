"""Streamlit dashboard: python -m streamlit run app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib

import clean_data
import eda
import recommend
from config import CLEAN_CSV, FEATURES, METRICS_PATH, MODEL_PATH, RANGES, TARGET

for _m in (clean_data, eda, recommend):  # pick up edits to src/ without restarting Streamlit
    importlib.reload(_m)

TEAL, RED, AMBER = "#3b8ea5", "#d95f5f", "#e0a030"
PRETTY = {f: f.replace("_", " ") for f in FEATURES}

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, label, input, button, textarea, h1, h2, h3, h4,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stTable"], .stTabs [role="tab"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.block-container {padding-top: 2.5rem;}
.hero {background: linear-gradient(120deg,#16384a,#3b8ea5); color:#fff; padding:1.4rem 1.8rem;
       border-radius:14px; margin-bottom:1rem;}
.hero h1 {margin:0; padding:0; font-size:1.9rem; font-weight:700; letter-spacing:-.02em; color:#fff;}
.hero p {margin:.4rem 0 0; opacity:.92; font-weight:400; font-size:1rem; color:#fff;}
.stTabs [role="tab"] {font-weight:600;}
div[data-testid="stMetric"] {background: rgba(59,142,165,.10); border-radius:10px; padding:.6rem .9rem;}
</style>""", unsafe_allow_html=True)


plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]  # first installed font wins
import plotly.io as pio
pio.templates["default"] = go.layout.Template(layout=dict(font=dict(family="Inter, Segoe UI, sans-serif")))
pio.templates.default = "plotly+default"


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_CSV)


@st.cache_resource
def load_bundle():
    return recommend.load_model()


@st.cache_data
def load_metrics(mtime: float):  # mtime in the cache key -> reloads when the model is retrained
    return json.loads(METRICS_PATH.read_text())


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def gauge(prob):
    color = RED if prob < .4 else AMBER if prob < .65 else TEAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Placement probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "rgba(217,95,95,.15)"},
                         {"range": [40, 65], "color": "rgba(224,160,48,.15)"},
                         {"range": [65, 100], "color": "rgba(59,142,165,.15)"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


if not (MODEL_PATH.exists() and CLEAN_CSV.exists()):
    st.error("Model/data not found. Run `python run_pipeline.py` first.")
    st.stop()

df, bundle, metrics = load_data(), load_bundle(), load_metrics(METRICS_PATH.stat().st_mtime)
if "n_train" not in metrics or "roc" not in metrics:
    st.error("Saved model files are from an older version. Run `python run_pipeline.py`, then refresh this page.")
    st.stop()
median = df[FEATURES].median().to_dict()
placed_avg = df[df[TARGET] == 1][FEATURES].mean()

st.markdown(f"""<div class="hero"><h1>🎓 Student Performance & Placement Prediction</h1>
<p>Data-driven placement forecasts, explanations and a personalised improvement plan —
powered by {metrics['best']} trained on {len(df):,} student records.</p></div>""", unsafe_allow_html=True)

tab_overview, tab_viz, tab_predict, tab_batch, tab_model = st.tabs(
    ["📊 Overview", "📈 Visualizations", "🔮 Predict & Improve", "📁 Batch Prediction", "🤖 Model"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    c = st.columns(5)
    c[0].metric("Students", f"{len(df):,}")
    c[1].metric("Placement rate", f"{df[TARGET].mean():.1%}")
    c[2].metric("Avg CGPA", f"{df.CGPA.mean():.2f}")
    c[3].metric("Avg attendance", f"{df.Attendance.mean():.1f}%")
    c[4].metric("Avg coding score", f"{df.Coding_Score.mean():.1f}")
    left, right = st.columns([1, 2])
    with left:
        show(eda.placement_overview(df))
    with right:
        st.subheader("What separates placed students?")
        prof = df.groupby(TARGET)[FEATURES].mean().rename(index={0: "Not Placed", 1: "Placed"}).T.round(2)
        prof["Difference"] = (prof["Placed"] - prof["Not Placed"]).round(2)
        st.dataframe(prof, width="stretch")
        imp = pd.Series(metrics["importance"]).sort_values()
        st.plotly_chart(px.bar(x=imp.values, y=[PRETTY[i] for i in imp.index], orientation="h",
                               color_discrete_sequence=[TEAL], title="Most influential factors")
                        .update_layout(height=300, xaxis_title="Relative importance", yaxis_title=""),
                        width="stretch")
    with st.expander("Browse dataset"):
        st.dataframe(df, width="stretch")

# ---------------------------------------------------------------- Visualizations
with tab_viz:
    st.caption("Filter the cohort to see how the patterns change.")
    f1, f2 = st.columns(2)
    cg = f1.slider("CGPA range", 0.0, 10.0, (float(df.CGPA.min()), float(df.CGPA.max())), 0.1)
    at = f2.slider("Attendance range (%)", 0.0, 100.0, (float(df.Attendance.min()), float(df.Attendance.max())), 1.0)
    d = df[df.CGPA.between(*cg) & df.Attendance.between(*at)]
    if len(d) < 20:
        st.warning("Too few students match these filters; widen the ranges.")
    else:
        st.write(f"**{len(d):,} students** · placement rate **{d[TARGET].mean():.1%}**")
        a, b = st.columns(2)
        with a:
            show(eda.box_by_status(d, "CGPA", "CGPA vs Placement"))
            show(eda.rate_by_bucket(d, "Internships", title="Internships vs Placement Rate"))
            show(eda.rate_by_bucket(d, "Backlogs", title="Backlogs vs Placement Rate"))
        with b:
            show(eda.cgpa_attendance_scatter(d))
            show(eda.box_by_status(d, "Coding_Score", "Coding Score vs Placement"))
            show(eda.box_by_status(d, "Aptitude_Score", "Aptitude Score vs Placement"))
        show(eda.correlation_heatmap(d))

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Enter student details")
    with st.form("student_form"):
        c1, c2, c3, c4 = st.columns(4)
        s = {
            "CGPA": c1.number_input("CGPA", *RANGES["CGPA"], 7.0, 0.1),
            "Attendance": c2.number_input("Attendance (%)", *RANGES["Attendance"], 80.0, 1.0),
            "Coding_Score": c3.number_input("Coding score", *RANGES["Coding_Score"], 55.0, 1.0),
            "Aptitude_Score": c4.number_input("Aptitude score", *RANGES["Aptitude_Score"], 60.0, 1.0),
            "Internships": c1.number_input("Internships", *RANGES["Internships"], 0),
            "Projects": c2.number_input("Projects", *RANGES["Projects"], 2),
            "Backlogs": c3.number_input("Backlogs", *RANGES["Backlogs"], 0),
            "Certifications": c4.number_input("Certifications", *RANGES["Certifications"], 1),
        }
        go_btn = st.form_submit_button("Predict placement", type="primary")

    if go_btn:
        st.session_state["student"] = s
    s = st.session_state.get("student")

    if s:
        label, prob = recommend.predict(bundle, s)
        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(gauge(prob), width="stretch")
            if label:
                st.success("✅ Likely to be **PLACED**")
            else:
                st.error("❌ Likely **NOT placed** — see the action plan")
        with r2:
            pct = recommend.percentiles(df, s)
            cats = [PRETTY[f] for f in FEATURES]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(r=[pct[f] for f in FEATURES] + [pct[FEATURES[0]]],
                                            theta=cats + [cats[0]], fill="toself", name="This student",
                                            line_color=TEAL))
            radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=300,
                                title="Standing within the cohort (percentile)",
                                margin=dict(l=40, r=40, t=50, b=20), showlegend=False)
            st.plotly_chart(radar, width="stretch")

        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("Why this prediction?")
            ex = recommend.explain(bundle, s, median)
            fig = go.Figure(go.Bar(
                x=ex.impact * 100, y=[f"{PRETTY[f]} ({v:g})" for f, v in zip(ex.feature, ex.value)],
                orientation="h", marker_color=[TEAL if i > 0 else RED for i in ex.impact]))
            fig.update_layout(height=340, xaxis_title="Effect on probability vs a typical student (% points)",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        with e2:
            st.subheader("Action plan")
            plan = recommend.action_plan(bundle, s)
            if not plan:
                st.info("This profile meets every benchmark. Keep it up!")
            for p in plan[:5]:
                sign = "≤" if p["feature"] == "Backlogs" else "≥"
                st.warning(f"**{PRETTY[p['feature']]}** — yours: {p['value']:g}, target {sign} {p['target']:g} "
                           f"(**{p['gain'] * 100:+.1f} pts** → {p['new_prob']:.0%})\n\n{p['suggestion']}")

        # Interactive what-if
        st.divider()
        st.subheader("🎛️ What-if simulator")
        st.caption("Adjust the sliders to see how changes would move the probability.")
        w = st.columns(4)
        wi = {}
        sig = hash(tuple(s[f] for f in FEATURES))  # new profile -> fresh sliders
        for i, f in enumerate(FEATURES):
            lo, hi = RANGES[f]
            step = 0.1 if f == "CGPA" else 1.0 if isinstance(lo, float) else 1
            wi[f] = w[i % 4].slider(PRETTY[f], lo, hi, type(lo)(s[f]), step, key=f"wi_{f}_{sig}")
        _, p_new = recommend.predict(bundle, wi)
        st.metric("Simulated placement probability", f"{p_new:.1%}", f"{(p_new - prob) * 100:+.1f} pts vs entered profile")
        st.caption("Predictions are statistical estimates from historical data, not guarantees.")
    else:
        st.info("Fill in the form and click **Predict placement**.")

# ---------------------------------------------------------------- Batch
with tab_batch:
    st.subheader("Score many students at once")
    st.write("Upload a CSV with columns: " + ", ".join(f"`{f}`" for f in FEATURES) + ". Extra columns (e.g. `Student_ID`) are kept.")
    template = pd.DataFrame([median]).round(1).astype({k: int for k in ["Internships", "Projects", "Backlogs", "Certifications"]})
    st.download_button("Download template CSV", template.to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        try:
            raw = pd.read_csv(up)
            missing = [f for f in FEATURES if f not in raw.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # Reuse the cleaning rules (range checks + median imputation)
                tmp = raw.copy()
                tmp[TARGET] = 0
                clean, _ = clean_data.clean(tmp)
                scored = clean.drop(columns=TARGET)
                scored["Placement_Probability"] = recommend.predict_many(bundle, scored).round(3)
                scored["Prediction"] = scored.Placement_Probability.ge(.5).map({True: "Placed", False: "Not Placed"})
                st.success(f"Scored {len(scored)} students — predicted placed: {(scored.Prediction == 'Placed').mean():.0%}")
                st.dataframe(scored.sort_values("Placement_Probability"), width="stretch")
                st.download_button("Download predictions", scored.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

# ---------------------------------------------------------------- Model
with tab_model:
    st.subheader(f"Best model: {metrics['best']}")
    st.caption(f"Each model was tuned with grid search + 5-fold CV on {metrics['n_train']} training students and "
               f"evaluated on {metrics['n_test']} unseen students. The winner was chosen by CV ROC-AUC.")
    res = pd.DataFrame(metrics["results"]).T
    st.dataframe(res.style.highlight_max(axis=0, color="#cfe8ef").format("{:.3f}"), width="stretch")
    st.caption("Best hyper-parameters: " + "; ".join(f"{k}: {v}" for k, v in metrics["best_params"].items()))
    m1, m2 = st.columns(2)
    with m1:
        fig = go.Figure()
        for name, r in metrics["roc"].items():
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=f"{name} (AUC {metrics['results'][name]['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
        fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=380, legend=dict(x=.4, y=.1))
        st.plotly_chart(fig, width="stretch")
    with m2:
        cm = metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Not Placed", "Placed"], y=["Not Placed", "Placed"],
                        color_continuous_scale="Blues", title=f"Confusion matrix — {metrics['best']}")
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with st.expander("Limitations & responsible use"):
        st.markdown("- The bundled dataset is **synthetic**; retrain on real institutional data before relying on results.\n"
                    "- Predictions show association, not causation; use them to guide support, not to exclude students.\n"
                    "- Factors such as communication skills or interview performance are not captured.")


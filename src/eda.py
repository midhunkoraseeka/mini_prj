"""Steps 3-4: EDA + visualisations. Figure builders return matplotlib figures (used by the dashboard too)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLEAN_CSV, FEATURES, FIG_DIR, TARGET

PALETTE = {0: "#d95f5f", 1: "#3b8ea5"}
LABELS = {0: "Not Placed", 1: "Placed"}
sns.set_theme(style="whitegrid")


def _with_label(df):
    d = df.copy()
    d["Status"] = d[TARGET].map(LABELS)
    return d


def _pal():
    return {LABELS[k]: v for k, v in PALETTE.items()}


def placement_overview(df):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    counts = df[TARGET].value_counts().reindex([1, 0])
    ax.pie(counts, labels=[LABELS[i] for i in counts.index], autopct="%1.1f%%", startangle=90,
           colors=[PALETTE[i] for i in counts.index], wedgeprops={"edgecolor": "white"})
    ax.set_title("Overall Placement Percentage")
    return fig


def box_by_status(df, col, title=None):
    d = _with_label(df)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.boxplot(data=d, x="Status", y=col, hue="Status", order=["Not Placed", "Placed"],
                palette=_pal(), legend=False, ax=ax)
    ax.set_title(title or f"{col} vs Placement")
    return fig


def rate_by_bucket(df, col, bins=None, title=None):
    d = df.copy()
    d["bucket"] = pd.cut(d[col], bins) if bins is not None else d[col]
    rate = d.groupby("bucket", observed=True)[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(rate.index.astype(str), rate.values, color=PALETTE[1])
    ax.set_ylabel("Placement rate (%)")
    ax.set_xlabel(col)
    ax.set_ylim(0, 100)
    ax.set_title(title or f"Placement rate by {col}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig


def cgpa_attendance_scatter(df):
    d = _with_label(df)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.scatterplot(data=d, x="Attendance", y="CGPA", hue="Status", palette=_pal(), alpha=.6, ax=ax)
    ax.set_title("Attendance vs CGPA (performance)")
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    return fig


def all_figures(df):
    return {
        "placement_overview": placement_overview(df),
        "cgpa_vs_placement": box_by_status(df, "CGPA"),
        "attendance_vs_performance": cgpa_attendance_scatter(df),
        "internship_vs_placement": rate_by_bucket(df, "Internships", title="Internships vs Placement Rate"),
        "coding_vs_placement": box_by_status(df, "Coding_Score", "Coding Score vs Placement"),
        "aptitude_vs_placement": box_by_status(df, "Aptitude_Score", "Aptitude Score vs Placement"),
        "backlogs_vs_placement": rate_by_bucket(df, "Backlogs", title="Backlogs vs Placement Rate"),
        "correlation": correlation_heatmap(df),
    }


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV)
    FIG_DIR.mkdir(exist_ok=True)
    print(df.describe().round(2).T)
    print("\nCorrelation with placement:\n", df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values().round(3))
    for name, fig in all_figures(df).items():
        fig.savefig(FIG_DIR / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved figures to {FIG_DIR}")

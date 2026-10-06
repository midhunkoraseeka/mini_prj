from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "student_placement_raw.csv"
CLEAN_CSV = ROOT / "data" / "student_placement_clean.csv"
MODEL_PATH = ROOT / "models" / "placement_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports"

TARGET = "Placed"
FEATURES = [
    "CGPA", "Attendance", "Internships", "Projects",
    "Coding_Score", "Aptitude_Score", "Backlogs", "Certifications",
]
# Valid ranges used for cleaning and for dashboard inputs: (min, max)
RANGES = {
    "CGPA": (0.0, 10.0),
    "Attendance": (0.0, 100.0),
    "Internships": (0, 5),
    "Projects": (0, 10),
    "Coding_Score": (0.0, 100.0),
    "Aptitude_Score": (0.0, 100.0),
    "Backlogs": (0, 10),
    "Certifications": (0, 10),
}

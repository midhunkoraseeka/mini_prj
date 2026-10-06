"""Create a synthetic student dataset (with realistic mess) when no real data is available.

Replace data/student_placement_raw.csv with a real dataset using the same columns to use your own data.
"""
import numpy as np
import pandas as pd

from config import RAW_CSV


def generate(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    cgpa = np.clip(rng.normal(7.2, 1.0, n), 4.0, 10.0).round(2)
    attendance = np.clip(rng.normal(80, 11, n), 40, 100).round(1)
    internships = np.clip(rng.poisson(0.9, n), 0, 4)
    projects = np.clip(rng.poisson(2.5, n), 0, 8)
    coding = np.clip(40 + 6 * (cgpa - 7) + 4 * internships + rng.normal(10, 15, n), 0, 100).round(1)
    aptitude = np.clip(55 + 5 * (cgpa - 7) + rng.normal(5, 14, n), 0, 100).round(1)
    backlogs = np.clip(rng.poisson(np.clip(1.2 - 0.6 * (cgpa - 6), 0.05, 3)), 0, 8)
    certs = np.clip(rng.poisson(1.5, n), 0, 8)

    z = (1.1 * (cgpa - 7) + 0.03 * (attendance - 80) + 0.55 * internships + 0.18 * projects
         + 0.04 * (coding - 55) + 0.02 * (aptitude - 60) - 0.7 * backlogs + 0.2 * certs
         + rng.normal(0, 0.8, n) - 0.2)
    placed = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "Student_ID": [f"S{i:04d}" for i in range(1, n + 1)],
        "CGPA": cgpa, "Attendance": attendance, "Internships": internships,
        "Projects": projects, "Coding_Score": coding, "Aptitude_Score": aptitude,
        "Backlogs": backlogs, "Certifications": certs,
        "Placed": np.where(placed == 1, "Yes", "No"),
    })

    # Inject mess for the cleaning step: missing values, duplicates, bad formatting, outliers.
    for col, frac in [("CGPA", .03), ("Attendance", .04), ("Coding_Score", .04), ("Aptitude_Score", .03)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "Placed"] = "  yes "
    df.loc[rng.choice(n, 15, replace=False), "Placed"] = "NO"
    df.loc[rng.choice(n, 5, replace=False), "CGPA"] = 85.0  # entered on a 100 scale
    df.loc[rng.choice(n, 4, replace=False), "Attendance"] = 130
    df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=3).reset_index(drop=True)


if __name__ == "__main__":
    RAW_CSV.parent.mkdir(exist_ok=True)
    generate().to_csv(RAW_CSV, index=False)
    print(f"Wrote {RAW_CSV}")

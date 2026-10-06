# Student Performance & Placement Prediction System

Predicts a student's placement outcome and probability from CGPA, attendance, internships, projects,
coding/aptitude scores, backlogs and certifications — then explains *why* and tells the student *what to fix first*.

## What makes it different
- **Explainable predictions:** per-student chart showing how each factor pushes the probability up or down vs a typical student.
- **Ranked action plan:** each weak area is simulated through the model, so suggestions are ordered by the real probability gain (e.g. "+6.3 pts").
- **What-if simulator:** sliders to see how changes would move the probability live.
- **Cohort percentile radar** and probability gauge.
- **Batch prediction:** upload a CSV of students, get scored results and a download (reuses the cleaning rules).
- **Tuned model comparison:** grid search + 5-fold CV for Logistic Regression, Decision Tree and Random Forest; ROC curves, confusion matrix, feature importance. The winner is chosen on CV, not on the test set.
- **Interactive EDA filters**, automated tests, and a limitations/responsible-use note.

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py                # generate data -> clean -> EDA charts -> tune & train
python -m streamlit run app.py        # dashboard
python -m pytest                      # tests
```

## Structure

| Path | Purpose |
|---|---|
| `src/generate_data.py` | Synthetic, deliberately messy dataset (swap in your own CSV with the same columns) |
| `src/clean_data.py` | Missing values, duplicates, label formatting, out-of-range fixes |
| `src/eda.py` | All EDA charts (also saved to `reports/`) |
| `src/train_model.py` | Grid-search tuning, comparison, saves best model + metrics |
| `src/recommend.py` | Prediction, explanation, action plan, percentiles |
| `app.py` | Streamlit dashboard (5 tabs) |
| `tests/` | pytest suite |

## Using real data
Place a CSV at `data/student_placement_raw.csv` with columns: `Student_ID, CGPA, Attendance, Internships, Projects,
Coding_Score, Aptitude_Score, Backlogs, Certifications, Placed (Yes/No)`, then re-run `python run_pipeline.py`.

## Note
The bundled dataset is **synthetic**, so patterns and metrics reflect how it was generated, not real students.
Predictions show association, not causation.

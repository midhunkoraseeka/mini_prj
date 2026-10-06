"""Step 5: tune, train & compare Logistic Regression, Decision Tree, Random Forest; save the best."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import CLEAN_CSV, FEATURES, METRICS_PATH, MODEL_PATH, TARGET

# name -> (pipeline, hyper-parameter grid searched with 5-fold CV)
SEARCH = {
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (
        Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
        {"clf__max_depth": [3, 4, 5, 7], "clf__min_samples_leaf": [5, 10, 20]}),
    "Random Forest": (
        Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
        {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [5, 10]}),
}


def _importance(model):
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    imp = imp / imp.sum()
    return dict(sorted(zip(FEATURES, map(float, imp)), key=lambda kv: -kv[1]))


def main():
    df = pd.read_csv(CLEAN_CSV)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    results, roc, best_params, tuned = {}, {}, {}, {}
    for name, (pipe, grid) in SEARCH.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        model = gs.best_estimator_
        pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        results[name] = {
            "cv_roc_auc": round(float(gs.best_score_), 4),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
        }
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc[name] = {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()}
        best_params[name] = {k.split("__")[1]: v for k, v in gs.best_params_.items()}
        tuned[name] = (model, confusion_matrix(y_te, pred).tolist())
        print(f"{name:20s}", results[name], best_params[name])

    best = max(results, key=lambda n: results[n]["cv_roc_auc"])  # select on CV, not on the test set
    print(f"\nBest model (by CV ROC-AUC): {best}")

    # Refit the winner (with its tuned parameters) on all data for deployment
    final = SEARCH[best][0].set_params(**{f"clf__{k}": v for k, v in best_params[best].items()}).fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best": best, "results": results, "best_params": best_params, "roc": roc,
        "confusion_matrix": tuned[best][1], "importance": _importance(final),
        "n_train": len(X_tr), "n_test": len(X_te),
    }, indent=2))
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()

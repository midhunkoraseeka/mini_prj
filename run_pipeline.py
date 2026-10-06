"""Run the whole pipeline: data -> clean -> EDA -> train."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

import clean_data, eda, generate_data, train_model
from config import CLEAN_CSV, FIG_DIR, RAW_CSV

if __name__ == "__main__":
    if not RAW_CSV.exists():
        RAW_CSV.parent.mkdir(exist_ok=True)
        generate_data.generate().to_csv(RAW_CSV, index=False)
        print("Generated synthetic raw dataset")
    df, log = clean_data.clean(pd.read_csv(RAW_CSV))
    df.to_csv(CLEAN_CSV, index=False)
    print("Cleaning log:", log)
    FIG_DIR.mkdir(exist_ok=True)
    for name, fig in eda.all_figures(df).items():
        fig.savefig(FIG_DIR / f"{name}.png", dpi=110, bbox_inches="tight")
    print("EDA figures saved")
    train_model.main()

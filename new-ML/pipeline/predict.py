import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from pipeline.config import (FEATURE_COLS, FAMILY_COL, MODELS_DIR, TABLES_DIR)

DOMAIN = {
    "L_mm": (3250.0, 8000.0),
    "b_mm": (250.0, 400.0),
    "h_mm": (325.0, 700.0),
    "fc_MPa": (25.0, 45.0),
    "rho_percent": (0.8, 2.0),
    "crack1_depth_mm": (50.0, 350.0),
    "crack2_depth_mm": (50.0, 350.0),
}
FAMILIES = ("FF", "SS")


class FrequencyPredictor:
    """Trained model + scaler with Ch. 3.10.3 domain enforcement."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, inputs):
        missing = [c for c in FEATURE_COLS + [FAMILY_COL] if c not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        if inputs[FAMILY_COL] not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        for col in FEATURE_COLS:
            lo, hi = DOMAIN[col]
            val = inputs[col]
            if not (lo <= val <= hi):
                raise ValueError(f"{col}={val} outside domain [{lo}, {hi}]")
        num = np.array([[inputs[c] for c in FEATURE_COLS]])
        Xs = self.scaler.transform(num)
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(Xs)).ravel()
        raise ValueError("model must expose predict()")


def save_artifacts(model, scaler, out_dir=MODELS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "best_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    meta = {"family_mode": "onehot", "features": FEATURE_COLS,
            "targets": [f"f{i}_hz" for i in range(1, 6)],
            "domain": DOMAIN, "families": list(FAMILIES)}
    (out_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def das_benchmark_table(pooled_r2, per_mode_r2, out_dir=TABLES_DIR):
    """Conceptual comparison vs Das (2023) for steel/aluminum beams."""
    rows = [
        {"Reference": "Das_2023", "Best_R2": 0.9878,
         "Note": "steel/aluminum beams, various BC (SVM, Puk kernel)"},
        {"Reference": "Das_2023_RandomForest", "Best_R2": 0.9888,
         "Note": "steel/aluminum beams, various BC"},
        {"Reference": "This_study_pooled", "Best_R2": float(pooled_r2),
         "Note": "fixed-fixed RC, FF/SS crack families"},
    ]
    for i, (idx, r2) in enumerate(per_mode_r2.items()):
        rows.append({"Reference": f"This_study_{idx}", "Best_R2": float(r2),
                     "Note": "per-mode"})
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "das_benchmark_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path

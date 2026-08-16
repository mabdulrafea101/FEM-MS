import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pipeline.config import (FEATURE_COLS, FAMILY_COL, TARGET_COLS,
                             DEV_SIZE, SEED)


def select_features(df):
    """Keep only the 8-field model matrix plus targets (leakage-controlled)."""
    keep = FEATURE_COLS + [FAMILY_COL] + TARGET_COLS
    return df[keep].copy()


def split_data(df, dev_size=DEV_SIZE, seed=SEED):
    """Random 800/200 split, stratified by family, seeded."""
    dev = pd.DataFrame()
    held = pd.DataFrame()
    for fam in ["FF", "SS"]:
        sub = df[df[FAMILY_COL] == fam]
        n_dev = dev_size // 2
        dev_part = sub.sample(n=n_dev, random_state=seed)
        dev = pd.concat([dev, dev_part])
        held = pd.concat([held, sub.drop(dev_part.index)])
    return dev, held


def make_xy(df):
    X = df[FEATURE_COLS + [FAMILY_COL]]
    y = df[TARGET_COLS]
    return X, y


class Preprocessor:
    """Scale 7 continuous features; encode family one-hot or keep native."""

    def __init__(self, family_mode="onehot", seed=SEED):
        assert family_mode in ("onehot", "native")
        self.family_mode = family_mode
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first", handle_unknown="ignore")

    def fit(self, X):
        self.scaler.fit(X[FEATURE_COLS])
        if self.family_mode == "onehot":
            self.encoder.fit(X[[FAMILY_COL]])
        return self

    def transform(self, X):
        num = self.scaler.transform(X[FEATURE_COLS])
        if self.family_mode == "onehot":
            fam = self.encoder.transform(X[[FAMILY_COL]]).toarray()
            num = np.hstack([num, fam])
        return {"num": num, "family": X[FAMILY_COL].reset_index(drop=True)}

    def fit_transform(self, X):
        return self.fit(X).transform(X)

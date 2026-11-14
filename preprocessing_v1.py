# Name: Srishanth Reddy Surakanti
# Class: CPT_S 437
# Task: Unified multi-site dataframe, missingness summary, and a simple v1 preprocessing pipeline
#
# Reuses Sai's loader (baseline_model.load_all_regions) so everyone reads the data the same way:
# '?' -> NaN, and 0 in chol/trestbps -> NaN.

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import baseline_model as bm

RESULTS_DIR = bm.RESULTS_DIR
LOCATION = "location"
# Features we treat as continuous (median impute + scale); everything else gets mode imputation
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Default cutoff: drop a feature if more than this fraction of the TRAINING rows are missing
MISSING_THRESHOLD = 0.50


# *** 1. UNIFIED DATAFRAME ***


# All four sites in one DataFrame: 13 features + `num` target + `location` column
def load_unified(data_dir=bm.DATA_DIR):
    df = bm.load_all_regions(data_dir)
    df = df.rename(columns={"region": LOCATION})
    return df


# *** 2. MISSINGNESS SUMMARY ***


# % missing per feature (rows) per location (columns), plus a pooled "All sites" column
def missing_table(df):
    feats = bm.FEATURES
    by_site = df.groupby(LOCATION)[feats].apply(lambda g: g.isna().mean() * 100).T
    by_site = by_site[[n for n in bm.REGIONS.values() if n in by_site.columns]]
    by_site["All sites"] = df[feats].isna().mean() * 100
    return by_site.round(1)


# Heatmap of the table above, annotated with the percentages
def plot_missing(table, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(table.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=20, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else "black")
    ax.set_title("Missing values by feature and hospital (%)")
    fig.colorbar(im, ax=ax, label="% missing")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# *** 3. V1 PREPROCESSING ***


# Drops features whose missing fraction in the data it is fit on exceeds `threshold`.
# Learning the drop list in fit() (not on the full dataset) keeps test data out of the decision.
class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=MISSING_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        frac = X.isna().mean()
        self.columns_ = [c for c in X.columns if frac[c] <= self.threshold]
        self.dropped_ = [c for c in X.columns if frac[c] > self.threshold]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.columns_]


# Simple v1 pipeline: drop >50% missing -> median (continuous) / mode (binary & categorical) -> standard scaling.
# Works on a DataFrame with the 13 feature columns. Fit on train splits only.
def build_v1_preprocessor(threshold=MISSING_THRESHOLD, features=None):
    features = features or bm.FEATURES
    cont = [c for c in CONTINUOUS if c in features]
    other = [c for c in features if c not in cont]

    def keep(cols):  # selects the surviving columns of a given group after the drop step
        return lambda X: [c for c in cols if c in X.columns]

    return Pipeline(
        [
            ("drop_missing", DropHighMissing(threshold)),
            (
                "impute_scale",
                ColumnTransformer(
                    [
                        ("cont", Pipeline([("impute", SimpleImputer(strategy="median")),
                                           ("scale", StandardScaler())]), keep(cont)),
                        ("disc", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                           ("scale", StandardScaler())]), keep(other)),
                    ],
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            ),
        ]
    )


# Convenience for quick exploration: fit on ALL rows and return a clean DataFrame (+ target and location).
# Do NOT use this to evaluate models -- fit the pipeline on training folds instead (see README).
def preprocess_all(df, threshold=MISSING_THRESHOLD):
    pre = build_v1_preprocessor(threshold)
    X = pre.fit_transform(df[bm.FEATURES])
    X[bm.TARGET] = df[bm.TARGET].to_numpy()
    X[LOCATION] = df[LOCATION].to_numpy()
    return X, pre.named_steps["drop_missing"].dropped_


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=MISSING_THRESHOLD)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_unified()
    print(f"Unified data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df[LOCATION].value_counts().to_string(), "\n")

    table = missing_table(df)
    table.to_csv(os.path.join(RESULTS_DIR, "missingness_by_site.csv"), index_label="feature")
    plot_missing(table, os.path.join(RESULTS_DIR, "missingness_heatmap.png"))
    print("% missing per feature:\n", table.to_string(), "\n")

    X, dropped = preprocess_all(df, args.threshold)
    X.to_csv(os.path.join(RESULTS_DIR, "preprocessed_v1.csv"), index=False)
    print(f"Dropped (> {args.threshold:.0%} missing, pooled): {dropped}")
    print(f"Preprocessed: {X.shape[0]} rows, {X.shape[1] - 2} features, 0 NaNs left: {not X.isna().any().any()}")


if __name__ == "__main__":
    main()

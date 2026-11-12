# Name: Sairaghav Gubba
# Date: Nov. 9, 2026,
# Class: CPT_S 437
# Task: Implementing baseline models (logistic regression, decision tree) and cross-validation scaffolding the rest of the team can reuse

# Import standard libraries
import argparse
import glob
import os

# Import third-party libraries
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Fixed random seed so fold splits and tree fits are reproducible run to run
SEED = 42

# Paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Maps the key used in each file name (processed.<key>.data)
REGIONS = {
    "cleveland": "Cleveland",
    "hungarian": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}
TARGET = "num"  # 0 = no heart disease, 1-4 = increasing severity
# All 13 attributes in the datasets
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
ZERO_MISSING = ["chol", "trestbps"]  # Value of 0 is marked for missing values


# Remove features from the model inputs (i.e. ca/thal) which are 90-99% missing outside Cleveland
def drop_features(names):
    for group in (NUMERIC, BINARY, CATEGORICAL, FEATURES):
        group[:] = [f for f in group if f not in names]


# *** DATA LOADING ***


# Return the path to one region's file, or None if it isn't there.
def find_region_file(data_dir, key):
    for pattern in (f"processed.{key}.data", f"processed_{key}.data"):
        hits = glob.glob(os.path.join(data_dir, pattern))
        if hits:
            return hits[0]
    return None


# Load one region's CSV and clean it into a consistent format
# '?' is how the files mark missing values, so read them as NaN
def load_region(path, region_name):
    df = pd.read_csv(path, na_values="?", skipinitialspace=True)

    # Fail if a file is missing a column we need
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {missing_cols}")

    # Keep only the columns we use and force everything to numbers
    # Any leftover non-numeric stuff becomes NaN instead of crashing later
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")

    # Treat impossible zeros as missing (see ZERO_MISSING)
    for col in ZERO_MISSING:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    df[TARGET] = df[TARGET].astype(int)

    # Tag every row with its region so we can group/split by site later
    df["region"] = region_name
    return df


# Load every region found in data_dir into a single DataFrame.
def load_all_regions(data_dir=DATA_DIR):
    frames = []
    for key, name in REGIONS.items():
        path = find_region_file(data_dir, key)
        if path is None:
            print(f"[skip] no file found for {name} in {data_dir}/")
            continue
        frames.append(load_region(path, name))
    if not frames:
        raise FileNotFoundError(f"No processed.<key>.data files found in {data_dir}/")
    # ignore_index=True gives the combined frame a clean 0..N-1 index
    return pd.concat(frames, ignore_index=True)


# Build the label array y for the chosen task.
#   binary: 1 if any heart disease (num > 0), else 0 (primary task)
#   multiclass: the raw severity num = 0..4 (secondary task)
def make_target(df, task):
    if task == "binary":
        return (df[TARGET] > 0).astype(int).to_numpy()
    return df[TARGET].to_numpy()


# Print record counts, disease rate and % missing per region.
# Only features with some missing data are shown, to keep the table narrow
def describe_data(df):
    print("\nRecords and missing values per region")
    print("-" * 70)
    summary = df.groupby("region").agg(
        n=(TARGET, "size"), pct_disease=(TARGET, lambda s: 100 * (s > 0).mean())
    )

    # Fraction of NaNs per feature per region, as a percentage
    miss = df[FEATURES].isna().groupby(df["region"]).mean() * 100
    miss = miss.loc[:, (miss > 0).any()].round(0)  # drop never-missing columns
    print(summary.round(1).join(miss.add_prefix("%miss_")).to_string())


# *** MODELS ***


# Build the preprocessing step that goes at the front of every model.
#   - Numeric: median imputation + 0/1 "was missing" flag per column, then optional standardization
#               (needed for linear regression, not trees)
#   - Binary: most-frequent imputation
#   - Categorical: most-frequent imputation, then one-hot encoding
def build_preprocessor(scale=True):
    numeric_steps = [
        (
            "impute",
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
        )
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    # ColumnTransformer applies a different pipeline to each feature group and concatenates the results into one feature matrix
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC),
            (
                "bin",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                BINARY,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


# Return {name: estimator} for every baseline model.
# The tuned models are wrapped in GridSearchCV. When one of these is passed to cross_validate, the grid search runs again inside each outer training split (makes evaluation nested)
def build_models(inner_folds=5, quick=False, n_jobs=-1):
    if quick:
        inner_folds = 3
    # Inner CV used by GridSearchCV to pick hyperparameters
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED)

    # Reference model: always predicts the most common class in its training data.
    # Any real model should beat this; if not, it has learned nothing
    majority = Pipeline(
        [
            ("prep", build_preprocessor(scale=False)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    # Logistic Regression. Grid parameters are written as "<step>__<param>", so GridSearchCV can reach inside the Pipeline
    #   - C: inverse regularization strength (small C = simpler model)
    #   - class_weight: "balanced" upweights the rarer class
    logreg = GridSearchCV(
        Pipeline(
            [
                ("prep", build_preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        ),
        param_grid=(
            {"model__C": [0.1, 1]}
            if quick
            else {
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
            }
        ),
        scoring="f1_macro",  # balances both classes and avoid rewarding "always yes"
        cv=inner_cv,
        n_jobs=n_jobs,
    )

    # Decision Tree:
    #   max_depth: how many splits deeps the tree can grow
    #   min_samples_leaf: minimum patients per lear (limits overfitting)
    #   criterion: impurity measure used to choose splits
    tree = GridSearchCV(
        Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),  # trees don't need scaling
                ("model", DecisionTreeClassifier(random_state=SEED)),
            ]
        ),
        param_grid=(
            {"model__max_depth": [3, 5]}
            if quick
            else {
                "model__max_depth": [2, 3, 4, 5, 6, 8, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__criterion": ["gini", "entropy"],
            }
        ),
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=n_jobs,
    )

    return {
        "Majority class": majority,
        "Logistic Regression": logreg,
        "Decision Tree": tree,
    }


# *** METRICS ***


# Return {metric name: scorer} for the chosen task.abs
# zero_division=0 avoids warnings when a model never predicts the positive class
# (e.g. the majority-class model), reporting 0 instead
def get_scoring(task):
    if task == "binary":
        return {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
        }

    # Multiclass: macro averages weight every severity level equally, so the
    # rare severe classes (num = 3,4) count as much as the common ones
    return {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }


# Collapse per-fold scores from cross_validate into mean and std
def _summarize(cv_out, scoring):
    row = {}
    for m in scoring:
        vals = cv_out[f"test_{m}"]
        row[m] = np.nanmean(vals)
        row[f"{m}_std"] = np.nanstd(vals)
    return row


# *** CROSS-VALIDATION SCAFFOLDING ***


# Stratified k-fold CV (the outer loop) for every model.
def run_kfold(X, y, models, scoring, n_folds, label):
    # Can't have more folds than members of the smallest class
    n_splits = min(n_folds, np.bincount(y).min())
    if n_splits < 2:
        print(f"  [skip] {label}: a class has fewer than 2 samples")
        return []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        # cross_validate clones and refits the whole pipeline on each fold.
        # error_score=np.nan records a failed fold as NaN instead of crashing
        out = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score=np.nan)
        rows.append({"setting": label, "model": name, **_summarize(out, scoring)})
    return rows


# Setting 1: k-fold CV separately inside each region
def run_within_region(df, y, models, scoring, n_folds):
    rows = []
    for region in df["region"].unique():
        mask = (df["region"] == region).to_numpy()
        rows += run_kfold(
            df.loc[mask, FEATURES],
            y[mask],
            models,
            scoring,
            n_folds,
            f"within: {region}",
        )
    return rows


# Setting 2: k-fold CV on all regions combined.
# Folds mix patients from every site, so this is the optimistic estimate
def run_pooled(df, y, models, scoring, n_folds):
    if df["region"].nunique() < 2:
        return []  # with one region this is identical to the within-region result
    return run_kfold(df[FEATURES], y, models, scoring, n_folds, "pooled")


# Setting 3: leave-one-cohort-out (LOCO)
# Each fold trains on every region but one and tests on the held-out region, measuring how well a model transfers to an unseen site
def run_loco(df, y, models, scoring):
    groups = df["region"].to_numpy()
    if len(np.unique(groups)) < 2:
        print("  [skip] LOCO needs at least two regions")
        return []
    logo = LeaveOneGroupOut()
    # Record which region each fold holds out, in the same order that cross_validate will iterate the folds
    held_out = [np.unique(groups[test])[0] for _, test in logo.split(df, y, groups)]
    rows = []
    for name, model in models.items():
        out = cross_validate(
            model,
            df[FEATURES],
            y,
            groups=groups,
            cv=logo,
            scoring=scoring,
            error_score=np.nan,
        )
        # One row per held-out region (not averaged), sinze the per-region differences are the point of this evaluation
        for i, region in enumerate(held_out):
            row = {"setting": f"LOCO test: {region}", "model": name}
            row.update({m: out[f"test_{m}"][i] for m in scoring})
            rows.append(row)
    return rows


# Setting 4: train on one region (default Cleveland), test on each other
# Mirrors how most published models are built (Cleveland only) and checks how well they hold up elsewhere
def run_train_one_test_others(df, y, models, scoring, source="Cleveland"):
    src = (df["region"] == source).to_numpy()
    if not src.any() or src.all():
        print(f"  [skip] train-on-{source} needs {source} plus another region")
        return []
    rows = []
    for name, model in models.items():
        # Fit once on the source region (grid search still tunes inside it)
        fitted = model.fit(df.loc[src, FEATURES], y[src])
        for region in df.loc[~src, "region"].unique():
            tgt = (df["region"] == region).to_numpy()
            row = {"setting": f"{source} -> {region}", "model": name}
            for m, scorer in scoring.items():
                # String metric names ("accuracy", "roc_auc") must be turned into scorer objects before they can be called directly
                scorer = scorer if callable(scorer) else get_scorer(scorer)
                try:
                    row[m] = scorer(fitted, df.loc[tgt, FEATURES], y[tgt])
                except ValueError:
                    # i.e. ROC-AUC is undefined if the target has one class
                    row[m] = np.nan
            rows.append(row)
    return rows


# *** REPORTING ***


# Print one results table (rows = setting x model, columns = metrics)
def print_table(rows, scoring, title):
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 70)
    table = pd.DataFrame(rows).set_index(["setting", "model"])[list(scoring)]
    # flush=True so tables appear immediately even when output is redirected
    print(table.round(3).to_string(), flush=True)


# Save grouped bar charts of recall and F1 per model and setting.
# Shows pooled CV next to each LOCO held-out region, which makes the drop in performance on unseen sites easy to see.
def plot_results(rows, scoring, task, out_path):
    df = pd.DataFrame(rows)
    df = df[df["setting"].str.startswith(("pooled", "LOCO"))]
    # e.g. if only one region loaded, fall back to whatever we have
    if df.empty:
        df = pd.DataFrame(rows)
    metrics = ["recall", "f1"] if task == "binary" else ["recall_macro", "f1_macro"]
    # dict.fromkeys keeps first-seen order while removing duplicates
    settings = list(dict.fromkeys(df["setting"]))
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 4.5), sharey=True
    )
    width = 0.8 / len(models)  # each setting's group of bars spans 0.8 units
    x = np.arange(len(settings))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for i, model in enumerate(models):
            vals = [
                df[(df.setting == s) & (df.model == model)][metric].mean()
                for s in settings
            ]
            # Offset each model's bar so the group is centred on the tick
            ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("LOCO test: ", "held out:\n") for s in settings], fontsize=9
        )
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Score")

    # One shared legend below the plots so it never covers a bar
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=9)
    fig.suptitle(f"Baseline models: pooled CV vs. leave-one-cohort-out ({task})")
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room at the bottom for the legend
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {out_path}")


# Print and save out-of-fold confusion matrices on the pooled data
# cross_val_predict gives each patient a prediction from a model that never saw them, so the matrix shows honest error counts
def report_pooled_confusion(df, y, models, task, n_folds, out_dir, tag=None):
    cv = StratifiedKFold(
        n_splits=min(n_folds, np.bincount(y).min()), shuffle=True, random_state=SEED
    )
    print("\nPooled out-of-fold confusion matrices (rows = true, cols = predicted)")
    print("-" * 70)
    matrices = {}
    for name in ("Logistic Regression", "Decision Tree"):
        pred = cross_val_predict(models[name], df[FEATURES], y, cv=cv)
        cm = confusion_matrix(y, pred)
        matrices[name] = cm
        print(f"{name}:\n{cm}")
        if task == "binary":
            # For a 2x2 matrix, ravel() returns counts in this fixed order
            tn, fp, fn, tp = cm.ravel()
            print(f"  missed diagnoses (FN) = {fn}, false alarms (FP) = {fp}")
        pd.DataFrame(cm).to_csv(
            os.path.join(
                out_dir, f"confusion_{name.replace(' ', '_').lower()}_{tag or task}.csv"
            )
        )
    plot_confusion_matrices(
        matrices,
        task,
        os.path.join(out_dir, f"confusion_{tag or task}.png"),
    )


# Draw the confusion matrices side by side as heatmaps.
# Cell shade = share of that true class (row %), so the rarer class is as readable
# as the common one; each cell is labelled with its count and row %.
def plot_confusion_matrices(matrices, task, out_path):
    from matplotlib.colors import LinearSegmentedColormap

    # One-hue sequential ramp: near-white for 0% up to dark blue for 100%
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    )
    n_classes = next(iter(matrices.values())).shape[0]
    if task == "binary":
        labels = ["No disease", "Disease"]
    else:
        labels = [str(i) for i in range(n_classes)]

    size = 4.2 if n_classes == 2 else 5.2
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(size * len(matrices) + 1, size), squeeze=False
    )
    for k, (ax, (name, cm)) in enumerate(zip(axes[0], matrices.items())):
        row_totals = cm.sum(axis=1, keepdims=True)
        share = np.divide(cm, row_totals, out=np.zeros(cm.shape), where=row_totals > 0)
        im = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)

        # White grid lines between cells (a thin gap so cells read as separate)
        ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(n_classes):
            for j in range(n_classes):
                # Light text on dark cells, dark text on light cells
                color = "white" if share[i, j] > 0.55 else "#1f1f1f"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{100 * share[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11 if n_classes == 2 else 9,
                )
        ax.set_xticks(range(n_classes), labels)
        # Panels share the same rows, so only the first one needs y labels
        ax.set_yticks(range(n_classes), labels if k == 0 else [])
        ax.tick_params(length=0)
        what = "" if task == "binary" else " severity (num)"
        ax.set_xlabel(f"Predicted{what}")
        if k == 0:
            ax.set_ylabel(f"True{what}")
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}\n(accuracy {acc:.3f}, n = {cm.sum()})", fontsize=11)

    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.03)
    cbar.set_label("Share of true class")
    cbar.outline.set_visible(False)
    fig.suptitle(f"Pooled out-of-fold confusion matrices ({task})")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# *** MAIN ***


def main():
    # Command-line options
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="folder with processed.*.data (default: <repo root>/data)",
    )
    parser.add_argument(
        "--out-dir",
        default=RESULTS_DIR,
        help="where results are written (default: <repo root>/results)",
    )
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument(
        "--folds", type=int, default=5, help="number of outer and inner CV folds"
    )
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        metavar="COL",
        help="features to exclude, e.g. ca thal",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="CPU cores for hyperparameter search (-1 = all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small hyperparameter grids for fast test runs",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}/ (task = {args.task})")
    df = load_all_regions(args.data_dir)
    y = make_target(df, args.task)

    # `tag` goes into output file names so runs with different options
    # (e.g. binary vs. binary_no_ca_thal) don't overwrite each other.
    tag = args.task
    if args.drop_features:
        unknown = set(args.drop_features) - set(FEATURES)
        if unknown:
            parser.error(f"unknown feature(s): {sorted(unknown)}")
        drop_features(args.drop_features)
        tag += "_no_" + "_".join(args.drop_features)
        print(f"Excluding features: {args.drop_features}")
    describe_data(df)

    # Build models and metrics (after any feature drops)
    models = build_models(inner_folds=args.folds, quick=args.quick, n_jobs=args.jobs)
    scoring = get_scoring(args.task)

    # Run the 4 evaluation settings
    # Print each table as soon as its stage finishes (nested CV takes a while).
    within = run_within_region(df, y, models, scoring, args.folds)
    print_table(within, scoring, f"Within-region {args.folds}-fold CV (mean)")
    pooled = run_pooled(df, y, models, scoring, args.folds)
    print_table(pooled, scoring, f"Pooled {args.folds}-fold CV (mean)")
    loco = run_loco(df, y, models, scoring)
    print_table(loco, scoring, "Leave-one-cohort-out (train on others, test on region)")
    transfer = run_train_one_test_others(df, y, models, scoring, "Cleveland")
    print_table(transfer, scoring, "Train on Cleveland only, test on each other region")

    # Save results
    all_rows = within + pooled + loco + transfer
    csv_path = os.path.join(args.out_dir, f"baseline_results_{tag}.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nSaved results -> {csv_path}")

    # Plot pooled + LOCO if available, otherwise everything we ran.
    plot_results(
        pooled + loco or all_rows,
        scoring,
        args.task,
        os.path.join(args.out_dir, f"baseline_{tag}.png"),
    )
    report_pooled_confusion(df, y, models, args.task, args.folds, args.out_dir, tag)

    # Show the tuned hyperparameters
    # Refit on all pooled data to see what the grid search settles on overall.
    print("\nHyperparameters chosen on pooled data")
    print("-" * 70)
    for name in ("Logistic Regression", "Decision Tree"):
        best = models[name].fit(df[FEATURES], y).best_params_
        print(f"{name}: {best}")


if __name__ == "__main__":
    main()

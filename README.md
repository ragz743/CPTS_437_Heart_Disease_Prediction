# CPTS 437 Heart Disease Prediction

**Multi-Region Cardiovascular Disease Prediction and Cross-Dataset Model Evaluation**

CptS 437 (Machine Learning) final project, Team 1, Washington State University.

Most published heart disease models are trained and tested on the Cleveland cohort alone, so their reported accuracy may not hold at other hospitals. This project builds supervised classifiers that predict whether a patient has heart disease from 13 routine clinical measurements. It then measures how well those models generalize to patients from geographically distinct sites, where much of the data is missing.

## Team

| Member | Responsibility |
| --- | --- |
| Srishanth Reddy Surakanti | Exploratory data analysis, multi-center missing-data handling, preprocessing pipeline |
| Sai Gubba | Baseline models (Logistic Regression, Decision Tree) and cross-validation scaffolding |
| Anh Duong | Ensemble and tree-based models (Random Forest, XGBoost) and hyperparameter optimization |
| Danbi Kim | Cross-dataset generalization testing (LOCO evaluation), error analysis, metric visualization |

## Dataset

[UCI Machine Learning Repository – Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). We use the standard 14-attribute processed files from four sites, 920 patients in total.

| Site | File | Patients | % with disease |
| --- | --- | --- | --- |
| Cleveland Clinic Foundation | `processed.cleveland.data` | 303 | 45.9 |
| Hungarian Institute of Cardiology, Budapest | `processed.hungarian.data` | 294 | 36.1 |
| University Hospitals, Zurich and Basel | `processed.switzerland.data` | 123 | 93.5 |
| VA Medical Center, Long Beach | `processed.va.data` | 200 | 74.5 |

Each file has a header row added for the attribute names. Missing values are marked `?`. The loader accepts either `processed.<region>.data` or `processed_<region>.data`.

### Attributes

| Attribute | Description |
| --- | --- |
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG: 0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment: 1 up, 2 flat, 3 down |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | 3 normal, 6 fixed defect, 7 reversible defect |
| `num` | **Target:** 0 = no disease, 1–4 = increasing severity |

The primary task is binary (`num > 0`). Predicting the severity level is a secondary task.

The header row in our data files spells the ECG column `restcg`. The code renames it to `restecg` when loading.

### Data quality notes

Cleveland is nearly complete. The other three sites are not:

- `ca` is missing for 96–99% of patients outside Cleveland.
- `thal` is missing for 42–90% of patients outside Cleveland.
- Switzerland records cholesterol as `0` for every patient. The code treats a `0` in `chol` or `trestbps` as missing.
- Roughly a quarter of VA patients are missing several vitals.

## Repository structure

```
CPTS_437_Heart_Disease_Prediction/
├── README.md
├── baseline_model.py        # baseline models + cross-validation scaffolding
├── data/
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   └── processed.va.data
└── results/                 # created automatically when the scripts run
```

## Setup

Requires Python 3.9+ and scikit-learn 1.2 or newer.

```bash
git clone https://github.com/ragz743/CPTS_437_Heart_Disease_Prediction.git
cd CPTS_437_Heart_Disease_Prediction
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```bash
python baseline_model.py                           # binary task: disease vs. no disease
python baseline_model.py --task multiclass         # severity: num = 0–4
python baseline_model.py --drop-features ca thal   # exclude the sparsest features
python baseline_model.py --quick                   # small search grids for a fast test run
python baseline_model.py --folds 10 --jobs 4       # 10 CV folds, limit to 4 CPU cores
```

Paths are resolved from the script's location, so it reads `data/` and writes `results/` correctly from any working directory. A full run takes a few minutes on a single core and is faster with more cores.

### Outputs

Each run writes the following to `results/`:

| File | Contents |
| --- | --- |
| `baseline_results_<tag>.csv` | Every metric for every model and evaluation setting |
| `baseline_<tag>.png` | Recall and F1 bar charts: pooled CV vs. each held-out region |
| `confusion_<model>_<tag>.csv` | Out-of-fold confusion matrix on the pooled data |

`<tag>` is the task name plus any dropped features (for example `binary_no_ca_thal`), so different runs don't overwrite each other. The script also prints each results table and the tuned hyperparameters to the console.

### Reusing the evaluation code

Other models can plug into the same evaluation by importing from `baseline_model.py`:

```python
from baseline_model import (
    load_all_regions,
    make_target,
    build_preprocessor,
    get_scoring,
    run_within_region,
    run_pooled,
    run_loco,
    run_train_one_test_others,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = load_all_regions()
y = make_target(df, "binary")
models = {
    "Random Forest": Pipeline(
        [
            ("prep", build_preprocessor(scale=False)),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
}
rows = run_loco(df, y, models, get_scoring("binary"))
```

## Methodology

**Preprocessing.** All preprocessing runs inside a scikit-learn `Pipeline` and is refit on each training split, so no information leaks from test data. The steps are:

- Median imputation for numeric features, with a flag column marking which values were missing.
- Most-frequent imputation for binary and categorical features.
- One-hot encoding for categorical features.
- Standardization for Logistic Regression only.

**Models.**

- **Majority class:** always predicts the most common training label. Any real model should beat it.
- **Logistic Regression:** tuned over regularization strength `C` and class weighting.
- **Decision Tree:** tuned over depth, minimum leaf size, and split criterion.

Hyperparameters are chosen with an inner `GridSearchCV` inside each outer training split (nested cross-validation), optimizing macro F1. Test folds never influence model selection.

**Evaluation settings.**

1. **Within-region:** stratified 5-fold CV inside each site.
2. **Pooled:** stratified 5-fold CV on all sites combined.
3. **Leave-one-cohort-out (LOCO):** train on three sites, test on the fourth.
4. **Cleveland only:** train on Cleveland and test on each other site, mirroring how most published models are built.

**Metrics.** Accuracy, precision, recall, F1 and ROC-AUC for the binary task. Macro-averaged precision, recall and F1 for the multiclass task. Recall (sensitivity) is prioritized because a missed diagnosis is costlier than a false alarm.

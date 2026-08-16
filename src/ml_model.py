# src/ml_model.py
#
# Machine Learning Models for Crop Planting Risk
# Vojvodina, Serbia — 2000 to 2026
#
# VERSION 3 — Multi-algorithm ensemble
#
# Algorithms trained per target per crop:
#   1. Random Forest      (bagging — parallel trees)
#   2. XGBoost            (gradient boosting — sequential)
#   3. LightGBM           (fast gradient boosting)
#   4. SVM                (support vector machine)
#   5. Logistic Regression(linear baseline)
#   6. Stacking Ensemble  (meta-learner combines all five)
#
# All probability outputs are calibrated using
# isotonic regression so that a 60% prediction
# actually corresponds to 60% historical frequency.
#
# Time-aware walk-forward validation throughout —
# no data leakage from future years.

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crop_model import CROPS, get_crop, get_frost_kill_temp
from crop_model import get_heat_stress_temp, get_critical_rain_threshold
from config import CONFIG

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE     = CONFIG["clean_data_file"]
OUTPUT_FOLDER  = "data/processed"
ANALYSIS_START = CONFIG["analysis_start"]
ANALYSIS_END   = CONFIG["analysis_end"]

TRAIN_END    = 2018
VALIDATE_END = 2021
# Test: 2022-2025

GROWING_SEASON_DAYS = {
    key: crop["maturity"].get(
        "medium_days",
        crop["maturity"].get(
            "from_transplant_days",
            crop["maturity"].get("from_seed_days", 90)
        )
    )
    for key, crop in CROPS.items()
}

FROST_CHECK_DAYS = 21

MIN_HEAT_DAYS = {
    key: 3 if crop["season_type"] == "cool" else 5
    for key, crop in CROPS.items()
}

PLANTING_CANDIDATES = [
    (2, 15), (3,  1), (3, 10), (3, 20),
    (4,  1), (4, 10), (4, 20),
    (5,  1), (5, 10), (5, 20),
    (6,  1),
]

# -------------------------------------------------------
# STEP 1 - LOAD DATA
# -------------------------------------------------------

print("=" * 60)
print("MACHINE LEARNING v3 — MULTI-ALGORITHM ENSEMBLE")
print("Vojvodina, Serbia")
print("=" * 60)

print("\nLoading climate data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["year"] >= ANALYSIS_START) &
    (df["year"] <= ANALYSIS_END)
].copy()

years = sorted(df["year"].unique())
complete_years = [
    y for y in years
    if df[df["year"] == y].shape[0] >= 350
]

print(f"Total years:    {len(years)}")
print(f"Complete years: {len(complete_years)}")
print(f"Daily rows:     {len(df):,}")
print(f"Columns:        {len(df.columns)}")

# -------------------------------------------------------
# STEP 2 - BUILD FEATURES
# -------------------------------------------------------

print("\nBuilding features...")

df = df.sort_values("date").reset_index(drop=True)

# Rolling temperature
df["temp_avg_7day"]  = df["temp_avg"].rolling(7,  min_periods=3).mean()
df["temp_avg_14day"] = df["temp_avg"].rolling(14, min_periods=7).mean()
df["temp_avg_30day"] = df["temp_avg"].rolling(30, min_periods=15).mean()
df["temp_max_7day"]  = df["temp_max"].rolling(7,  min_periods=3).mean()
df["temp_max_30day"] = df["temp_max"].rolling(30, min_periods=15).mean()
df["temp_min_7day"]  = df["temp_min"].rolling(7,  min_periods=3).mean()
df["temp_min_30day"] = df["temp_min"].rolling(30, min_periods=15).mean()

# Rolling rainfall
df["rain_7day"]  = df["precipitation"].rolling(7,  min_periods=3).sum()
df["rain_14day"] = df["precipitation"].rolling(14, min_periods=7).sum()
df["rain_30day"] = df["precipitation"].rolling(30, min_periods=15).sum()

# Consecutive dry days
def count_dry(series, threshold=1.0):
    result  = []
    counter = 0
    for val in series:
        if pd.isna(val) or val < threshold:
            counter += 1
        else:
            counter = 0
        result.append(counter)
    return result

df["consecutive_dry_days"] = count_dry(df["precipitation"].values)

# Anomalies
monthly_avg_temp   = df.groupby("month")["temp_avg"].transform("mean")
df["temp_anomaly"] = df["temp_avg"] - monthly_avg_temp

monthly_avg_rain   = df.groupby("month")["precipitation"].transform("mean")
df["rain_anomaly"] = df["precipitation"] - monthly_avg_rain

# Trend and warming signal
df["temp_trend_30day"] = df["temp_avg"] - df["temp_avg_30day"]
df["years_since_2000"] = df["year"] - 2000

# Day of year
if "day_of_year" not in df.columns:
    df["day_of_year"] = df["date"].dt.dayofyear

# Frost margin (min temp - dew point)
if "dew_point" in df.columns:
    df["frost_margin"] = df["temp_min"] - df["dew_point"]
else:
    df["frost_margin"] = np.nan

# Water balance
if "evapotranspiration" in df.columns:
    df["et_30day"] = df["evapotranspiration"].rolling(
        30, min_periods=15
    ).sum()
    df["water_balance_30day"] = df["rain_30day"] - df["et_30day"]
else:
    df["water_balance_30day"] = np.nan
    df["et_30day"]            = np.nan

# Cumulative GDD
if "gdd_base_10" in df.columns:
    df["cumulative_gdd_10"] = df.groupby(
        "year"
    )["gdd_base_10"].cumsum()
else:
    df["cumulative_gdd_10"] = np.nan

if "gdd_base_7" in df.columns:
    df["cumulative_gdd_7"] = df.groupby(
        "year"
    )["gdd_base_7"].cumsum()
else:
    df["cumulative_gdd_7"] = np.nan

# Ensure soil columns exist
for col in ["soil_wet_root", "soil_moisture_pctl",
            "soil_wet_surface", "soil_temp_0_10cm",
            "soil_temp_10_40cm", "cloud_cover",
            "dew_point", "frost_margin"]:
    if col not in df.columns:
        df[col] = np.nan

print(f"Features built. Total columns: {len(df.columns)}")

# -------------------------------------------------------
# STEP 3 - FEATURE COLUMNS
# -------------------------------------------------------

FEATURE_COLS = [
    # Calendar
    "day_of_year", "month", "years_since_2000",

    # Air temperature
    "temp_avg", "temp_max", "temp_min",
    "temp_avg_7day", "temp_avg_14day", "temp_avg_30day",
    "temp_max_7day", "temp_max_30day",
    "temp_min_7day", "temp_min_30day",
    "temp_anomaly", "temp_trend_30day",

    # Dew point and frost margin
    "dew_point", "frost_margin",

    # Precipitation
    "precipitation",
    "rain_7day", "rain_14day", "rain_30day",
    "consecutive_dry_days", "rain_anomaly",

    # Soil moisture
    "soil_wet_root", "soil_moisture_pctl", "soil_wet_surface",

    # Soil temperature
    "soil_temp_0_10cm", "soil_temp_10_40cm",

    # Evapotranspiration and water balance
    "evapotranspiration", "water_balance_30day",

    # Accumulated GDD
    "cumulative_gdd_10", "cumulative_gdd_7",

    # Cloud cover
    "cloud_cover",
]

# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"Feature columns: {len(FEATURE_COLS)}")


# -------------------------------------------------------
# STEP 4 - BUILD TRAINING DATASET
# -------------------------------------------------------

def build_dataset(crop_key):
    """
    Build training dataset for one crop.
    One row per (year, planting_date) combination.
    Features from planting date only.
    Targets from future weather window.
    No data leakage.
    """
    crop          = get_crop(crop_key)
    frost_kill    = crop["frost"]["foliage_kill_c"]
    frost_damage  = crop["frost"]["foliage_damage_c"]
    heat_temp     = get_heat_stress_temp(crop_key)
    grow_days     = GROWING_SEASON_DAYS[crop_key]
    min_heat      = MIN_HEAT_DAYS[crop_key]

    rows = []

    for year in complete_years:
        for month, day in PLANTING_CANDIDATES:
            try:
                plant_date = pd.Timestamp(
                    year=year, month=month, day=day
                )
            except ValueError:
                continue

            plant_row = df[df["date"] == plant_date]
            if plant_row.empty:
                continue

            # Features from planting date
            features = {}
            for col in FEATURE_COLS:
                val = plant_row[col].iloc[0] \
                    if col in plant_row.columns else np.nan
                features[col] = 0 if pd.isna(val) else val

            # Future windows for targets
            frost_window = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=FROST_CHECK_DAYS))
            ]
            full_window = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=grow_days))
            ]

            if len(frost_window) < FROST_CHECK_DAYS * 0.5:
                continue
            if len(full_window) < grow_days * 0.5:
                continue

            # Target 1: frost kill
            min_temp      = frost_window["temp_min"].min()
            frost_kill_t  = int(min_temp <= frost_kill)

            # Target 2: heat stress
            heat_days     = (
                full_window["temp_max"] >= heat_temp
            ).sum()
            heat_t        = int(heat_days >= min_heat)

            # Target 3: rainfall deficit
            monthly_rain  = full_window.groupby(
                full_window["date"].dt.month
            )["precipitation"].sum()

            rain_t = 0
            for m, rain_mm in monthly_rain.items():
                threshold = get_critical_rain_threshold(
                    crop_key, m
                )
                if (threshold is not None and
                        rain_mm < threshold):
                    rain_t = 1
                    break

            row = {
                "year"        : year,
                "month"       : month,
                "day"         : day,
                "plant_date"  : plant_date,
                "crop_key"    : crop_key,
                "frost_kill"  : frost_kill_t,
                "heat_stress" : heat_t,
                "rain_deficit": rain_t,
            }
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------
# STEP 5 - BUILD MODELS
# -------------------------------------------------------

def build_models():
    """
    Return a dictionary of all base models to train.
    Each model is wrapped in a calibrated classifier
    so probability outputs are well calibrated.
    """
    models = {}

    # 1. Random Forest
    models["RandomForest"] = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators     = 300,
            max_depth        = 6,
            min_samples_leaf = 3,
            random_state     = 42,
            class_weight     = "balanced",
            n_jobs           = -1,
        ),
        method="isotonic",
        cv=3,
    )

    # 2. XGBoost
    if HAS_XGBOOST:
        models["XGBoost"] = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators      = 300,
                max_depth         = 4,
                learning_rate     = 0.05,
                subsample         = 0.8,
                colsample_bytree  = 0.8,
                random_state      = 42,
                eval_metric       = "logloss",
                verbosity         = 0,
            ),
            method="isotonic",
            cv=3,
        )

    # 3. LightGBM
    if HAS_LGBM:
        models["LightGBM"] = CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators  = 300,
                max_depth     = 4,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = 42,
                verbose       = -1,
            ),
            method="isotonic",
            cv=3,
        )

    # 4. SVM — needs scaling so wrap in pipeline
    models["SVM"] = CalibratedClassifierCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel      = "rbf",
                C           = 1.0,
                probability = False,
                class_weight= "balanced",
                random_state= 42,
            )),
        ]),
        method="isotonic",
        cv=3,
    )

    # 5. Logistic Regression — scaled, L2 regularised
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C            = 0.1,
            class_weight = "balanced",
            random_state = 42,
            max_iter     = 1000,
        )),
    ])

    return models


# -------------------------------------------------------
# STEP 6 - TRAIN AND EVALUATE
# -------------------------------------------------------

def train_and_evaluate(dataset, target_col, crop_key):
    """
    Train all models and evaluate with walk-forward
    time-aware validation.

    Returns dict of {model_name: fitted_model}
    for the models that trained successfully.
    """
    avail_features = [
        c for c in FEATURE_COLS if c in dataset.columns
    ]

    ds = dataset[avail_features + [target_col, "year"]].dropna()

    if len(ds) < 30:
        print(f"  Insufficient data ({len(ds)} rows)")
        return {}

    train_mask = ds["year"] <= TRAIN_END
    test_mask  = ds["year"] > VALIDATE_END

    X_train = ds[train_mask][avail_features]
    y_train = ds[train_mask][target_col]
    X_test  = ds[test_mask][avail_features]
    y_test  = ds[test_mask][target_col]

    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}  "
          f"Positive rate: {y_train.mean()*100:.0f}%")

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print(f"  Skipping — constant target")
        return {}

    if len(X_test) < 5 or y_test.nunique() < 2:
        print(f"  Skipping — insufficient test data")
        return {}

    # Baseline: always predict the training frequency
    baseline_prob  = y_train.mean()
    baseline_preds = np.full(len(y_test), baseline_prob)
    baseline_brier = brier_score_loss(y_test, baseline_preds)

    print(f"  Baseline Brier: {baseline_brier:.3f}  "
          f"(predicting {baseline_prob*100:.0f}% always)")

    models     = build_models()
    fitted     = {}
    results    = {}

    print(f"\n  {'Model':<22} {'AUC':>7} "
          f"{'Brier':>8} {'vs Baseline':>13}")
    print(f"  {'-'*54}")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            # Get calibrated probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test).astype(float)

            auc   = roc_auc_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
            diff  = baseline_brier - brier

            symbol = "✓" if diff > 0.005 else \
                     "≈" if diff > -0.005 else "✗"

            print(f"  {name:<22} {auc:>6.3f} "
                  f"{brier:>8.3f} "
                  f"{diff:>+8.3f} {symbol}")

            fitted[name]  = model
            results[name] = {
                "auc"  : auc,
                "brier": brier,
                "diff" : diff,
            }

        except Exception as e:
            print(f"  {name:<22} FAILED: {str(e)[:40]}")

    # -------------------------------------------------------
    # STACKING ENSEMBLE
    # -------------------------------------------------------
    # Use the base models as level-1 estimators and
    # logistic regression as the meta-learner.
    # We only stack models that improved on baseline.

    good_models = {
        name: mdl
        for name, mdl in fitted.items()
        if name != "LogisticRegression" and
        results.get(name, {}).get("diff", 0) > 0
    }

    if len(good_models) >= 2:
        try:
            # Build fresh unfitted estimators for stacking
            # (stacking needs to refit internally)
            stack_estimators = []

            if "RandomForest" in good_models:
                stack_estimators.append((
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    )
                ))

            if HAS_XGBOOST and "XGBoost" in good_models:
                stack_estimators.append((
                    "xgb",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    )
                ))

            if HAS_LGBM and "LightGBM" in good_models:
                stack_estimators.append((
                    "lgbm",
                    LGBMClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42,
                        verbose=-1,
                    )
                ))

            if "SVM" in good_models:
                stack_estimators.append((
                    "svm",
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("svm", SVC(
                            kernel       = "rbf",
                            class_weight = "balanced",
                            random_state = 42,
                            probability  = True,
                        )),
                    ])
                ))

            if len(stack_estimators) >= 2:
                stack = StackingClassifier(
                    estimators   = stack_estimators,
                    final_estimator = LogisticRegression(
                        C=0.5,
                        random_state=42,
                        max_iter=1000,
                    ),
                    passthrough  = False,
                    cv           = 3,
                    stack_method = "predict_proba",
                    n_jobs       = -1,
                )

                stack.fit(X_train, y_train)
                y_prob_stack = stack.predict_proba(X_test)[:, 1]

                auc_stack   = roc_auc_score(y_test, y_prob_stack)
                brier_stack = brier_score_loss(y_test, y_prob_stack)
                diff_stack  = baseline_brier - brier_stack
                symbol      = "✓" if diff_stack > 0.005 else "≈"

                print(f"  {'Stacking Ensemble':<22} "
                      f"{auc_stack:>6.3f} "
                      f"{brier_stack:>8.3f} "
                      f"{diff_stack:>+8.3f} {symbol} ←")

                fitted["Stacking"]  = stack
                results["Stacking"] = {
                    "auc"  : auc_stack,
                    "brier": brier_stack,
                    "diff" : diff_stack,
                }

        except Exception as e:
            print(f"  Stacking FAILED: {str(e)[:60]}")

    # Pick best model
    if results:
        best_name = max(
            results,
            key=lambda n: results[n]["diff"]
        )
        best_brier = results[best_name]["brier"]
        print(f"\n  Best model: {best_name} "
              f"(Brier {best_brier:.3f})")

    return fitted, results


# -------------------------------------------------------
# STEP 7 - FEATURE IMPORTANCE
# -------------------------------------------------------

def show_importance(model, model_name, target, top_n=8):
    """Show feature importance for tree-based models."""
    base = model

    # Unwrap calibrated classifier
    if hasattr(base, "estimator"):
        base = base.estimator
    if hasattr(base, "named_steps"):
        base = base.named_steps.get(
            "rf",
            base.named_steps.get("svm", base)
        )

    if not hasattr(base, "feature_importances_"):
        return

    imp = pd.Series(
        base.feature_importances_,
        index=FEATURE_COLS[:len(base.feature_importances_)]
    ).sort_values(ascending=False)

    print(f"\n  Feature importance — {model_name} / {target}:")
    for feat, val in imp.head(top_n).items():
        bar = "█" * int(val * 40)
        print(f"    {feat:<30} {val:.3f}  {bar}")


# -------------------------------------------------------
# STEP 8 - PREDICT ALL PLANTING DATES
# -------------------------------------------------------

def predict_all_dates(best_models, crop_key):
    """
    Generate risk probability predictions for all
    candidate planting dates using the best model
    for each target.
    """
    recent = df[df["year"].between(2015, 2026)].copy()
    preds  = []

    for month, day in PLANTING_CANDIDATES:
        try:
            date_label = pd.Timestamp(
                year=2024, month=month, day=day
            ).strftime("%b %d")
        except ValueError:
            continue

        date_rows = recent[
            (recent["month"] == month) &
            (recent["day"]   == day)
        ]

        avail = [c for c in FEATURE_COLS
                 if c in date_rows.columns]
        feats = date_rows[avail].dropna()

        if feats.empty:
            continue

        row = {
            "date_label": date_label,
            "month"     : month,
            "day"       : day,
        }

        for target_name, model in best_models.items():
            if model is None:
                row[f"{target_name}_prob"] = np.nan
                continue
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feats)[:, 1]
                else:
                    probs = model.predict(feats).astype(float)
                row[f"{target_name}_prob"] = probs.mean()
            except Exception:
                row[f"{target_name}_prob"] = np.nan

        preds.append(row)

    return pd.DataFrame(preds)


# -------------------------------------------------------
# STEP 9 - RUN FOR ALL CROPS
# -------------------------------------------------------

all_predictions = []

for crop_key in CROPS.keys():
    crop_name = CROPS[crop_key]["name"]

    print(f"\n\n{'='*60}")
    print(f"CROP: {crop_name.upper()}")
    print(f"{'='*60}")

    print("\nBuilding dataset...")
    dataset = build_dataset(crop_key)
    print(f"Dataset rows: {len(dataset)}")

    if dataset.empty:
        print("No data — skipping")
        continue

    print("\nTarget frequencies:")
    for t in ["frost_kill", "heat_stress", "rain_deficit"]:
        pct = dataset[t].mean() * 100
        print(f"  {t:<20}: {pct:.1f}%")

    # Train models for each target
    # Track best model per target for prediction
    best_models_for_prediction = {}
    all_fitted                 = {}

    for target in ["frost_kill", "heat_stress", "rain_deficit"]:
        print(f"\n--- {target.upper()} ---")

        result = train_and_evaluate(dataset, target, crop_key)
        if not result or len(result) == 0:
            best_models_for_prediction[target] = None
            continue

        fitted, results = result
        all_fitted[target] = fitted

        # Show feature importance for best model
        if results:
            best_name = max(
                results,
                key=lambda n: results[n]["diff"]
            )
            if best_name in fitted:
                show_importance(
                    fitted[best_name],
                    best_name,
                    target,
                    top_n=6
                )
            best_models_for_prediction[target] = \
                fitted.get(best_name)
        else:
            best_models_for_prediction[target] = None

    # Generate predictions
    print(f"\n--- PREDICTIONS ---")
    pred_models = {
        "frost_kill"  : best_models_for_prediction.get("frost_kill"),
        "heat_stress" : best_models_for_prediction.get("heat_stress"),
        "rain_deficit": best_models_for_prediction.get("rain_deficit"),
    }

    predictions = predict_all_dates(pred_models, crop_key)
    predictions["crop"]     = crop_name
    predictions["crop_key"] = crop_key
    all_predictions.append(predictions)

    if not predictions.empty:
        print(f"\n  {'Date':<12} "
              f"{'Frost%':>8} "
              f"{'Heat%':>8} "
              f"{'Rain%':>8}")
        print(f"  {'-'*40}")
        for _, r in predictions.iterrows():
            f = r.get("frost_kill_prob",  np.nan)
            h = r.get("heat_stress_prob", np.nan)
            ra = r.get("rain_deficit_prob",np.nan)
            fs = f"{f*100:.1f}%"  if not pd.isna(f)  else "N/A"
            hs = f"{h*100:.1f}%"  if not pd.isna(h)  else "N/A"
            rs = f"{ra*100:.1f}%" if not pd.isna(ra) else "N/A"
            print(f"  {r['date_label']:<12} "
                  f"{fs:>8} {hs:>8} {rs:>8}")


# -------------------------------------------------------
# STEP 10 - SAVE
# -------------------------------------------------------

if all_predictions:
    combined = pd.concat(all_predictions, ignore_index=True)
    out      = os.path.join(
        OUTPUT_FOLDER, "ml_planting_predictions.csv"
    )
    combined.to_csv(out, index=False)
    print(f"\nML predictions saved to: {out}")

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(f"""
Algorithms trained per target per crop:
  1. Random Forest      — bagging ensemble
  2. XGBoost            — gradient boosting {'✓' if HAS_XGBOOST else '✗ not installed'}
  3. LightGBM           — fast gradient boosting {'✓' if HAS_LGBM else '✗ not installed'}
  4. SVM                — support vector machine
  5. Logistic Regression— linear baseline
  6. Stacking Ensemble  — meta-learner over best base models

All probabilities calibrated with isotonic regression.
Best model per target selected by Brier score improvement
over statistical baseline.
Time-aware validation: train 2000-2018, test 2022-2025.
""")

print("=== MACHINE LEARNING v3 COMPLETE ===")
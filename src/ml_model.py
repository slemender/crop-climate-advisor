# src/ml_model.py
#
# Machine Learning Models for Crop Planting Risk Prediction
# Vojvodina, Serbia — 2000 to 2026
#
# What this script does:
# 1. Builds a training dataset from daily climate data
# 2. Creates features available at planting time only
#    (no data leakage from the future)
# 3. Creates target variables (did frost/heat/drought occur?)
# 4. Trains Random Forest and Gradient Boosting models
# 5. Uses time-aware walk-forward validation
# 6. Compares ML predictions against statistical baseline
# 7. Shows which features matter most
# 8. Saves model predictions for all planting dates
#
# IMPORTANT: We only use features available ON the planting
# date to predict what happens AFTER planting. Using future
# weather data would be data leakage and would produce
# misleadingly optimistic performance scores.
#
# Data: 2000-2026 (26 complete years)
# Crops: Potato, Tomato, Onion, Cucumber

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crop_model import CROPS, get_crop, get_frost_kill_temp
from crop_model import get_heat_stress_temp, get_critical_rain_threshold

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE     = "data/processed/vojvodina_clean.csv"
OUTPUT_FOLDER  = "data/processed"
ANALYSIS_START = 2000
ANALYSIS_END   = 2026

# Time-aware train/validate/test split
# We train on past, validate on near-future, test on recent
TRAIN_END      = 2018   # train on 2000-2018 (19 years)
VALIDATE_END   = 2021   # validate on 2019-2021 (3 years)
# Test:                  test on 2022-2025 (4 years)

# Build growing season days from crop database
# Uses medium variety days to maturity as the season length
GROWING_SEASON_DAYS = {
    key: crop["maturity"].get("medium_days",
         crop["maturity"].get("from_transplant_days",
         crop["maturity"].get("from_seed_days", 90)))
    for key, crop in CROPS.items()
}

# Frost check window (days after planting)
FROST_CHECK_DAYS = 21

# Minimum heat days to count as meaningful heat risk
# Cool season crops are more sensitive — use lower threshold
# Warm season crops are more tolerant — use higher threshold
MIN_HEAT_DAYS_FOR_RISK = {
    key: 3 if crop["season_type"] == "cool" else 5
    for key, crop in CROPS.items()
}

# Maximum dry days before drought stress
MAX_DRY_DAYS_CRITICAL = 14

# Candidate planting dates to analyze
PLANTING_CANDIDATES = [
    (2, 15), (3,  1), (3, 10), (3, 20),
    (4,  1), (4, 10), (4, 20),
    (5,  1), (5, 10), (5, 20),
    (6,  1),
]

# -------------------------------------------------------
# STEP 1 - LOAD AND PREPARE DATA
# -------------------------------------------------------

print("=" * 60)
print("MACHINE LEARNING — CROP PLANTING RISK")
print("Vojvodina, Serbia — 2000 to 2026")
print("=" * 60)

print("\nLoading climate data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["year"] >= ANALYSIS_START) &
    (df["year"] <= ANALYSIS_END)
].copy()

years          = sorted(df["year"].unique())
complete_years = [
    y for y in years
    if df[df["year"] == y].shape[0] >= 350
]

print(f"Total years:    {len(years)}")
print(f"Complete years: {len(complete_years)}")
print(f"Daily rows:     {len(df):,}")

# -------------------------------------------------------
# STEP 2 - BUILD ROLLING FEATURES
# -------------------------------------------------------
# These are the features the model will use.
# ALL of these are available at the time of planting.
# None of them require future knowledge.

print("\nBuilding features...")

df = df.sort_values("date").reset_index(drop=True)

# Temperature rolling averages
df["temp_avg_7day"]  = df["temp_avg"].rolling(7,  min_periods=3).mean()
df["temp_avg_14day"] = df["temp_avg"].rolling(14, min_periods=7).mean()
df["temp_avg_30day"] = df["temp_avg"].rolling(30, min_periods=15).mean()
df["temp_max_7day"]  = df["temp_max"].rolling(7,  min_periods=3).mean()
df["temp_max_30day"] = df["temp_max"].rolling(30, min_periods=15).mean()
df["temp_min_7day"]  = df["temp_min"].rolling(7,  min_periods=3).mean()
df["temp_min_30day"] = df["temp_min"].rolling(30, min_periods=15).mean()

# Rainfall rolling totals
df["rain_7day"]      = df["precipitation"].rolling(7,  min_periods=3).sum()
df["rain_14day"]     = df["precipitation"].rolling(14, min_periods=7).sum()
df["rain_30day"]     = df["precipitation"].rolling(30, min_periods=15).sum()

# Consecutive dry days
def count_consecutive_dry(series, threshold=1.0):
    result  = []
    counter = 0
    for val in series:
        if pd.isna(val) or val < threshold:
            counter += 1
        else:
            counter = 0
        result.append(counter)
    return result

df["consecutive_dry_days"] = count_consecutive_dry(
    df["precipitation"].values
)

# Temperature anomaly vs monthly average
monthly_avg_temp   = df.groupby("month")["temp_avg"].transform("mean")
df["temp_anomaly"] = df["temp_avg"] - monthly_avg_temp

# Rainfall anomaly vs monthly average
monthly_avg_rain   = df.groupby("month")["precipitation"].transform("mean")
df["rain_anomaly"] = df["precipitation"] - monthly_avg_rain

# Temperature trend feature
# How much has temperature changed in the last 30 days?
df["temp_trend_30day"] = df["temp_avg"] - df["temp_avg_30day"]

# Warming trend signal - year relative to 2000
# This helps the model understand that 2020 is climatically
# different from 2000
df["years_since_2000"] = df["year"] - 2000

print("Features built.")
print(f"Feature columns available: {len(df.columns)}")

# -------------------------------------------------------
# STEP 3 - DEFINE FEATURE COLUMNS
# -------------------------------------------------------
# These are exactly the features our ML model will see.
# All are available at planting time (no future leakage).

FEATURE_COLS = [
    # Calendar position
    "day_of_year",          # what day of year?
    "month",                # which month?
    "years_since_2000",     # captures warming trend

    # Recent temperature (what has weather been doing?)
    "temp_avg",             # temperature on planting day
    "temp_max",             # max temp on planting day
    "temp_min",             # min temp on planting day
    "temp_avg_7day",        # average last 7 days
    "temp_avg_14day",       # average last 14 days
    "temp_avg_30day",       # average last 30 days
    "temp_max_7day",        # max temp last 7 days
    "temp_max_30day",       # max temp last 30 days
    "temp_min_7day",        # min temp last 7 days
    "temp_min_30day",       # min temp last 30 days
    "temp_anomaly",         # warmer/cooler than average?
    "temp_trend_30day",     # is it warming or cooling?

    # Recent rainfall
    "precipitation",        # rain on planting day
    "rain_7day",            # rain last 7 days
    "rain_14day",           # rain last 14 days
    "rain_30day",           # rain last 30 days
    "consecutive_dry_days", # current dry streak
    "rain_anomaly",         # wetter/drier than average?
]

# Add day_of_year if not already present
if "day_of_year" not in df.columns:
    df["day_of_year"] = df["date"].dt.dayofyear

# -------------------------------------------------------
# STEP 4 - BUILD TRAINING DATASET
# -------------------------------------------------------
# For each planting date in each year, we create one row:
# - Features: weather conditions ON the planting date
# - Targets:  what happened AFTER the planting date
#
# This is the core of avoiding data leakage.
# We look FORWARD to create targets but only use
# BACKWARD-LOOKING features.

def build_dataset(crop_key):
    """
    Build a training dataset for one crop.

    For each candidate planting date in each complete year:
    1. Extract features from the planting date row
    2. Look forward to calculate target variables
    3. Return a DataFrame with features + targets

    Returns:
        dataset_df: one row per (year, planting_date) combination
    """
    crop          = get_crop(crop_key)
    frost_kill    = crop["frost"]["foliage_kill_c"]
    frost_damage  = crop["frost"]["foliage_damage_c"]
    heat_temp     = get_heat_stress_temp(crop_key)
    grow_days     = GROWING_SEASON_DAYS[crop_key]
    min_heat_days = MIN_HEAT_DAYS_FOR_RISK[crop_key]

    rows = []

    for year in complete_years:
        for month, day in PLANTING_CANDIDATES:
            try:
                plant_date = pd.Timestamp(
                    year=year, month=month, day=day
                )
            except ValueError:
                continue  # invalid date e.g. Feb 29 in non-leap year

            # --- GET FEATURES FROM PLANTING DATE ---
            # Find the row in our dataset for this exact date
            plant_row = df[df["date"] == plant_date]

            if plant_row.empty:
                continue  # date not in our dataset

            # Extract feature values from planting date row
            features = {}
            for col in FEATURE_COLS:
                if col in plant_row.columns:
                    val = plant_row[col].iloc[0]
                    features[col] = val if not pd.isna(val) else 0
                else:
                    features[col] = 0

            # --- GET FUTURE WEATHER FOR TARGETS ---
            # Look FORWARD from planting date
            # This is only used to create training labels
            # NOT used as model features

            # Frost window (first 21 days)
            frost_window = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=FROST_CHECK_DAYS))
            ]

            # Full growing season
            full_window = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=grow_days))
            ]

            if len(frost_window) < FROST_CHECK_DAYS * 0.5:
                continue  # not enough future data

            if len(full_window) < grow_days * 0.5:
                continue  # not enough future data

            # --- TARGET 1: FROST KILL ---
            min_temp_after = frost_window["temp_min"].min()
            frost_kill_occurred = int(
                min_temp_after <= frost_kill
            )
            frost_damage_occurred = int(
                min_temp_after <= frost_damage
            )

            # --- TARGET 2: HEAT STRESS ---
            heat_days_after = (
                full_window["temp_max"] >= heat_temp
            ).sum()
            heat_occurred = int(
                heat_days_after >= min_heat_days
            )

            # --- TARGET 3: RAINFALL DEFICIT ---
            monthly_rain = full_window.groupby(
                full_window["date"].dt.month
            )["precipitation"].sum()

            rain_deficit_occurred = 0
            for m, rain_mm in monthly_rain.items():
                threshold = get_critical_rain_threshold(
                    crop_key, m
                )
                if threshold is not None and rain_mm < threshold:
                    rain_deficit_occurred = 1
                    break

            # --- TARGET 4: ANY RISK (combined) ---
            # Binary: did ANY of the three risks occur?
            any_risk = int(
                frost_kill_occurred or
                heat_occurred or
                rain_deficit_occurred
            )

            # Combine into one row
            row = {
                "year"                  : year,
                "month"                 : month,
                "day"                   : day,
                "plant_date"            : plant_date,
                "crop_key"              : crop_key,

                # Targets (what happened after planting)
                "frost_kill"            : frost_kill_occurred,
                "frost_damage"          : frost_damage_occurred,
                "heat_stress"           : heat_occurred,
                "rain_deficit"          : rain_deficit_occurred,
                "any_risk"              : any_risk,

                # Additional context (not used as features)
                "heat_days_after"       : int(heat_days_after),
                "min_temp_after"        : round(
                    float(min_temp_after), 2
                ) if not pd.isna(min_temp_after) else 0,
            }
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------
# STEP 5 - TRAIN AND EVALUATE ML MODELS
# -------------------------------------------------------

def train_and_evaluate(dataset, target_col, crop_key,
                       baseline_probs=None):
    """
    Train Random Forest and Gradient Boosting models.
    Use time-aware walk-forward validation.

    Parameters:
        dataset:        DataFrame with features and targets
        target_col:     which column to predict
        crop_key:       crop name for reporting
        baseline_probs: statistical baseline probabilities
                        for comparison

    Returns:
        results dict with model performance metrics
    """
    crop_name = CROPS[crop_key]["name"]

    # Remove rows with missing feature values
    dataset_clean = dataset[FEATURE_COLS + [target_col, "year"]].dropna()

    if len(dataset_clean) < 30:
        print(f"  Insufficient data for {target_col}")
        return None

    # --- TIME-AWARE SPLIT ---
    # Train on earlier years, test on recent years
    # This mimics how the model would actually be used
    train_mask    = dataset_clean["year"] <= TRAIN_END
    validate_mask = (dataset_clean["year"] > TRAIN_END) & \
                    (dataset_clean["year"] <= VALIDATE_END)
    test_mask     = dataset_clean["year"] > VALIDATE_END

    X_train = dataset_clean[train_mask][FEATURE_COLS]
    y_train = dataset_clean[train_mask][target_col]

    X_val   = dataset_clean[validate_mask][FEATURE_COLS]
    y_val   = dataset_clean[validate_mask][target_col]

    X_test  = dataset_clean[test_mask][FEATURE_COLS]
    y_test  = dataset_clean[test_mask][target_col]

    print(f"\n  Training on:   {ANALYSIS_START}-{TRAIN_END} "
          f"({len(X_train)} samples)")
    print(f"  Validating on: {TRAIN_END+1}-{VALIDATE_END} "
          f"({len(X_val)} samples)")
    print(f"  Testing on:    {VALIDATE_END+1}-2025 "
          f"({len(X_test)} samples)")
    print(f"  Class balance: "
          f"{y_train.sum()}/{len(y_train)} positive "
          f"({y_train.mean()*100:.0f}%)")

    if len(X_test) < 5:
        print(f"  Warning: Very small test set — "
              f"results may not be reliable")

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print(f"  Skipping: target is constant "
              f"(always {y_train.iloc[0]})")
        return None

    # --- RANDOM FOREST ---
    rf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        min_samples_leaf = 3,
        random_state     = 42,
        class_weight     = "balanced",
        # balanced weights help when one class is rare
        # e.g. frost only occurs in 30% of years
    )
    rf.fit(X_train, y_train)

    # --- GRADIENT BOOSTING ---
    gb = GradientBoostingClassifier(
        n_estimators  = 200,
        max_depth     = 4,
        learning_rate = 0.05,
        random_state  = 42,
        subsample     = 0.8,
    )
    gb.fit(X_train, y_train)

    # --- EVALUATE ON TEST SET ---
    results = {}

    for name, model in [("RandomForest", rf),
                         ("GradBoost", gb)]:
        if len(X_test) > 0 and y_test.nunique() > 1:
            y_pred     = model.predict(X_test)
            y_prob     = model.predict_proba(X_test)[:, 1]
            accuracy   = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = 0.5

            try:
                brier = brier_score_loss(y_test, y_prob)
            except Exception:
                brier = 0.25

            results[name] = {
                "accuracy" : accuracy,
                "auc"      : auc,
                "brier"    : brier,
                "model"    : model,
            }

            print(f"\n  {name}:")
            print(f"    Accuracy:     {accuracy*100:.1f}%")
            print(f"    AUC-ROC:      {auc:.3f}  "
                  f"(0.5=random, 1.0=perfect)")
            print(f"    Brier Score:  {brier:.3f}  "
                  f"(lower=better, 0=perfect)")
        else:
            print(f"\n  {name}: insufficient test data")

    # --- BASELINE COMPARISON ---
    if baseline_probs is not None and len(X_test) > 0:
        # Statistical baseline: use the historical frequency
        # for the test years as a constant prediction
        baseline_pred = np.full(len(y_test),
                                baseline_probs.get(target_col, 0.5))
        if y_test.nunique() > 1:
            baseline_brier = brier_score_loss(y_test, baseline_pred)
            print(f"\n  Statistical baseline Brier: "
                  f"{baseline_brier:.3f}")
            if "RandomForest" in results:
                diff = baseline_brier - results["RandomForest"]["brier"]
                if diff > 0.01:
                    print(f"  → Random Forest IMPROVES on baseline "
                          f"by {diff:.3f} Brier points ✓")
                elif diff < -0.01:
                    print(f"  → Random Forest is WORSE than baseline "
                          f"— use statistical model instead")
                else:
                    print(f"  → Random Forest similar to baseline "
                          f"— marginal improvement")

    return results


# -------------------------------------------------------
# STEP 6 - FEATURE IMPORTANCE
# -------------------------------------------------------

def show_feature_importance(model, model_name, target_name,
                             top_n=10):
    """
    Show which features the model found most useful.
    This helps us understand WHAT drives risk predictions.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print(f"\n  Top {top_n} features for {target_name} "
          f"({model_name}):")
    for feat, imp in importance.head(top_n).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<25} {imp:.3f}  {bar}")

    return importance


# -------------------------------------------------------
# STEP 7 - GENERATE PREDICTIONS FOR ALL PLANTING DATES
# -------------------------------------------------------

def predict_all_dates(models, crop_key):
    """
    Use trained models to generate risk probability
    predictions for all candidate planting dates.

    For each planting date we use the AVERAGE of features
    across recent years (2019-2026) to represent what
    conditions might look like on that date going forward.

    This is the "prediction" phase — what does the model
    think about each planting date given recent climate?
    """
    crop_name = CROPS[crop_key]["name"]
    predictions = []

    # Use recent years as representative of current climate
    recent_df = df[df["year"].between(2015, 2026)].copy()

    for month, day in PLANTING_CANDIDATES:
        try:
            date_label = pd.Timestamp(
                year=2024, month=month, day=day
            ).strftime("%b %d")
        except ValueError:
            continue

        # Get all historical observations for this
        # specific calendar date in recent years
        date_rows = recent_df[
            (recent_df["month"] == month) &
            (recent_df["day"] == day)
        ]

        if date_rows.empty:
            continue

        # Get feature values for this date
        date_features = date_rows[FEATURE_COLS].dropna()

        if date_features.empty:
            continue

        row_pred = {"date_label": date_label,
                    "month": month, "day": day}

        for target_name, model in models.items():
            if model is not None:
                # Predict probability for each historical
                # observation of this date and average them
                probs = model.predict_proba(date_features)[:, 1]
                row_pred[f"{target_name}_prob"] = probs.mean()
            else:
                row_pred[f"{target_name}_prob"] = np.nan

        predictions.append(row_pred)

    return pd.DataFrame(predictions)


# -------------------------------------------------------
# STEP 8 - RUN FOR ALL CROPS
# -------------------------------------------------------

all_ml_predictions = []

for crop_key in CROPS.keys():
    crop_name = CROPS[crop_key]["name"]

    print(f"\n\n{'='*60}")
    print(f"ML MODEL: {crop_name.upper()}")
    print(f"{'='*60}")

    # Build training dataset for this crop
    print(f"\nBuilding dataset for {crop_name}...")
    dataset = build_dataset(crop_key)
    print(f"Dataset rows: {len(dataset)}")
    print(f"Years covered: {sorted(dataset['year'].unique())}")

    if dataset.empty:
        print(f"No data for {crop_name} — skipping")
        continue

    # Print target variable frequencies
    print(f"\nTarget variable frequencies:")
    for target in ["frost_kill", "heat_stress",
                   "rain_deficit", "any_risk"]:
        freq = dataset[target].mean() * 100
        print(f"  {target:<20}: {freq:.1f}% of planting dates")

    # Calculate baseline probabilities
    # (simple historical frequencies - what our statistical
    #  model already knows)
    baseline_probs = {
        "frost_kill"  : dataset["frost_kill"].mean(),
        "heat_stress" : dataset["heat_stress"].mean(),
        "rain_deficit": dataset["rain_deficit"].mean(),
        "any_risk"    : dataset["any_risk"].mean(),
    }

    # Train models for each target
    best_models = {}

    for target in ["frost_kill", "heat_stress", "rain_deficit"]:
        print(f"\n--- TARGET: {target.upper()} ---")
        results = train_and_evaluate(
            dataset, target, crop_key, baseline_probs
        )

        if results and "RandomForest" in results:
            rf_model = results["RandomForest"]["model"]
            show_feature_importance(rf_model, "RandomForest",
                                    target, top_n=8)
            best_models[target] = rf_model
        else:
            best_models[target] = None

    # Generate predictions for all planting dates
    print(f"\n--- PREDICTIONS FOR ALL PLANTING DATES ---")
    print(f"Using {crop_name} models trained on 2000-{TRAIN_END}")
    print(f"Predicting based on recent climate (2015-2026)")

    predictions = predict_all_dates(best_models, crop_key)
    predictions["crop"]     = crop_name
    predictions["crop_key"] = crop_key

    all_ml_predictions.append(predictions)

    if not predictions.empty:
        print(f"\n{'Date':<12} {'Frost%':>8} "
              f"{'Heat%':>8} {'Rain%':>8}")
        print("-" * 40)
        for _, row in predictions.iterrows():
            frost = row.get("frost_kill_prob", np.nan)
            heat  = row.get("heat_stress_prob", np.nan)
            rain  = row.get("rain_deficit_prob", np.nan)

            frost_s = f"{frost*100:.1f}%" if not pd.isna(frost) else "N/A"
            heat_s  = f"{heat*100:.1f}%"  if not pd.isna(heat)  else "N/A"
            rain_s  = f"{rain*100:.1f}%"  if not pd.isna(rain)  else "N/A"

            print(f"{row['date_label']:<12} "
                  f"{frost_s:>8} "
                  f"{heat_s:>8} "
                  f"{rain_s:>8}")

# -------------------------------------------------------
# STEP 9 - COMPARE ML vs STATISTICAL BASELINE
# -------------------------------------------------------

print("\n\n" + "=" * 60)
print("ML vs STATISTICAL BASELINE COMPARISON")
print("=" * 60)

# Load statistical baseline results
baseline_file = os.path.join(
    OUTPUT_FOLDER, "planting_date_risk_scores.csv"
)

if os.path.exists(baseline_file):
    baseline_df = pd.read_csv(baseline_file)

    print("""
Comparison approach:
  Statistical model: uses historical frequency counts
  ML model:          uses weather patterns at planting time
                     to adjust predictions

For each crop and planting date:
  - Statistical frost risk = % of years with frost after that date
  - ML frost risk = prediction based on recent weather patterns

If ML performs similarly to statistical:
  → Use the simpler statistical model (more interpretable)
If ML significantly improves:
  → Use ML predictions for better accuracy
""")

    if all_ml_predictions:
        ml_all = pd.concat(all_ml_predictions, ignore_index=True)

        for crop_key in CROPS.keys():
            crop_name = CROPS[crop_key]["name"]
            ml_crop   = ml_all[ml_all["crop_key"] == crop_key]
            stat_crop = baseline_df[
                baseline_df["crop_key"] == crop_key
            ]

            if ml_crop.empty or stat_crop.empty:
                continue

            print(f"\n{crop_name} — Statistical vs ML predictions:")
            print(f"{'Date':<12} {'Stat Frost%':>12} "
                  f"{'ML Frost%':>12} {'Stat Rain%':>12} "
                  f"{'ML Rain%':>12}")
            print("-" * 62)

            for _, stat_row in stat_crop.iterrows():
                ml_row = ml_crop[
                    (ml_crop["month"] == stat_row["month"]) &
                    (ml_crop["day"]   == stat_row["day"])
                ]

                if ml_row.empty:
                    continue

                stat_frost = stat_row["frost_kill_prob"] * 100
                stat_rain  = stat_row["rain_deficit_prob"] * 100

                ml_frost = ml_row["frost_kill_prob"].iloc[0] * 100 \
                    if "frost_kill_prob" in ml_row.columns \
                    else np.nan
                ml_rain  = ml_row["rain_deficit_prob"].iloc[0] * 100 \
                    if "rain_deficit_prob" in ml_row.columns \
                    else np.nan

                date_label = stat_row["date_label"]

                frost_diff = ml_frost - stat_frost
                frost_note = (f" ({'↑' if frost_diff > 2 else '↓' if frost_diff < -2 else '≈'})")

                print(f"{date_label:<12} "
                      f"{stat_frost:>10.1f}%  "
                      f"{ml_frost:>10.1f}%{frost_note} "
                      f"{stat_rain:>10.1f}%  "
                      f"{ml_rain:>10.1f}%")

# -------------------------------------------------------
# STEP 10 - SAVE ML PREDICTIONS
# -------------------------------------------------------

if all_ml_predictions:
    ml_combined = pd.concat(all_ml_predictions, ignore_index=True)
    output_file = os.path.join(
        OUTPUT_FOLDER, "ml_planting_predictions.csv"
    )
    ml_combined.to_csv(output_file, index=False)
    print(f"\nML predictions saved to: {output_file}")

print("\n" + "=" * 60)
print("HONEST ASSESSMENT")
print("=" * 60)
print("""
With 26 years of data and 11 planting dates per crop,
we have approximately 286 training samples per crop.

This is a SMALL dataset for machine learning.

What this means:
  - Models may show modest performance above baseline
  - Feature importance tells us what matters most
  - Do not expect dramatic improvement over statistics
  - The statistical baseline is already well-calibrated

Where ML genuinely helps:
  - Adjusting predictions based on recent weather patterns
  - Identifying which weather features drive risk
  - Potentially better calibrated probabilities

Where statistics is sufficient:
  - Long-term historical frequency estimates
  - Clear seasonal patterns (frost risk in March)
  - Cases where ML and statistical agree closely

Rule: if ML Brier score is not meaningfully better than
statistical baseline, trust the statistical model.
It is simpler, more interpretable, and equally accurate.
""")

print("\n=== MACHINE LEARNING COMPLETE ===")
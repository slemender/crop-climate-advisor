# src/ml_model.py
#
# Machine Learning Models for Crop Planting Risk Prediction
# Vojvodina, Serbia — 2000 to 2026
#
# VERSION 2 — Expanded feature set using new NASA POWER
# variables: soil moisture, dew point, GDD, ET, cloud cover
#
# What this script does:
# 1. Builds training dataset from daily climate data
# 2. Creates features available at planting time only
# 3. Creates target variables (frost/heat/drought occurred?)
# 4. Trains Random Forest and Gradient Boosting models
# 5. Uses time-aware walk-forward validation
# 6. Compares ML against statistical baseline
# 7. Shows which features matter most
# 8. Saves predictions for all planting dates

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss
)

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

MIN_HEAT_DAYS_FOR_RISK = {
    key: 3 if crop["season_type"] == "cool" else 5
    for key, crop in CROPS.items()
}

MAX_DRY_DAYS_CRITICAL = 14

PLANTING_CANDIDATES = [
    (10,  1), (10, 15),
    (11,  1), (11, 15),
    (12,  1),
    (2,  15), (3,   1), (3,  10), (3,  20),
    (4,   1), (4,  10), (4,  20),
    (5,   1), (5,  10), (5,  20),
    (6,   1),
]

# -------------------------------------------------------
# STEP 1 - LOAD DATA
# -------------------------------------------------------

print("=" * 60)
print("MACHINE LEARNING — CROP PLANTING RISK v2")
print("Vojvodina, Serbia — Expanded Feature Set")
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
print(f"Columns:        {len(df.columns)}")

# -------------------------------------------------------
# STEP 2 - BUILD ALL FEATURES
# -------------------------------------------------------

print("\nBuilding features...")

df = df.sort_values("date").reset_index(drop=True)

# --- Rolling temperature ---
df["temp_avg_7day"]  = df["temp_avg"].rolling(7,  min_periods=3).mean()
df["temp_avg_14day"] = df["temp_avg"].rolling(14, min_periods=7).mean()
df["temp_avg_30day"] = df["temp_avg"].rolling(30, min_periods=15).mean()
df["temp_max_7day"]  = df["temp_max"].rolling(7,  min_periods=3).mean()
df["temp_max_30day"] = df["temp_max"].rolling(30, min_periods=15).mean()
df["temp_min_7day"]  = df["temp_min"].rolling(7,  min_periods=3).mean()
df["temp_min_30day"] = df["temp_min"].rolling(30, min_periods=15).mean()

# --- Rolling precipitation ---
df["rain_7day"]  = df["precipitation"].rolling(7,  min_periods=3).sum()
df["rain_14day"] = df["precipitation"].rolling(14, min_periods=7).sum()
df["rain_30day"] = df["precipitation"].rolling(30, min_periods=15).sum()

# --- Consecutive dry days ---
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

# --- Anomalies ---
monthly_avg_temp   = df.groupby("month")["temp_avg"].transform("mean")
df["temp_anomaly"] = df["temp_avg"] - monthly_avg_temp

monthly_avg_rain   = df.groupby("month")["precipitation"].transform("mean")
df["rain_anomaly"] = df["precipitation"] - monthly_avg_rain

# --- Temperature trend ---
df["temp_trend_30day"] = df["temp_avg"] - df["temp_avg_30day"]

# --- Warming trend signal ---
df["years_since_2000"] = df["year"] - 2000

# --- Day of year ---
if "day_of_year" not in df.columns:
    df["day_of_year"] = df["date"].dt.dayofyear

# --- NEW: Frost margin ---
# Distance between min temp and dew point
# Small margin = higher frost risk
if "dew_point" in df.columns:
    df["frost_margin"] = df["temp_min"] - df["dew_point"]
else:
    df["frost_margin"] = np.nan

# --- NEW: Water balance ---
if "evapotranspiration" in df.columns:
    df["et_30day"] = df["evapotranspiration"].rolling(
        30, min_periods=15
    ).sum()
    df["water_balance_30day"] = df["rain_30day"] - df["et_30day"]
else:
    df["water_balance_30day"] = np.nan
    df["et_30day"]            = np.nan

# --- NEW: Cumulative GDD since Jan 1 ---
if "gdd_base_10" in df.columns:
    df["cumulative_gdd_10"] = df.groupby("year")["gdd_base_10"].cumsum()
else:
    df["cumulative_gdd_10"] = np.nan

if "gdd_base_7" in df.columns:
    df["cumulative_gdd_7"] = df.groupby("year")["gdd_base_7"].cumsum()
else:
    df["cumulative_gdd_7"] = np.nan

# --- NEW: Ensure soil columns exist ---
for col in ["soil_wet_root", "soil_moisture_pctl",
            "soil_wet_surface", "soil_temp_0_10cm",
            "soil_temp_10_40cm", "cloud_cover",
            "dew_point", "frost_margin"]:
    if col not in df.columns:
        df[col] = np.nan

print(f"Features built. Total columns: {len(df.columns)}")

# -------------------------------------------------------
# STEP 3 - DEFINE FEATURE COLUMNS
# -------------------------------------------------------

FEATURE_COLS = [
    # Calendar
    "day_of_year",
    "month",
    "years_since_2000",

    # Air temperature
    "temp_avg",
    "temp_max",
    "temp_min",
    "temp_avg_7day",
    "temp_avg_14day",
    "temp_avg_30day",
    "temp_max_7day",
    "temp_max_30day",
    "temp_min_7day",
    "temp_min_30day",
    "temp_anomaly",
    "temp_trend_30day",

    # Dew point
    "dew_point",
    "frost_margin",

    # Precipitation
    "precipitation",
    "rain_7day",
    "rain_14day",
    "rain_30day",
    "consecutive_dry_days",
    "rain_anomaly",

    # Soil moisture (NEW)
    "soil_wet_root",
    "soil_moisture_pctl",
    "soil_wet_surface",

    # Soil temperature (NEW)
    "soil_temp_0_10cm",
    "soil_temp_10_40cm",

    # Evapotranspiration (NEW)
    "evapotranspiration",
    "water_balance_30day",

    # Accumulated GDD (NEW)
    "cumulative_gdd_10",
    "cumulative_gdd_7",

    # Cloud cover (NEW)
    "cloud_cover",
]

# Only keep columns that actually exist in the dataframe
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"Feature columns available: {len(FEATURE_COLS)}")

# -------------------------------------------------------
# STEP 4 - BUILD TRAINING DATASET
# -------------------------------------------------------

def build_dataset(crop_key):
    """
    Build training dataset for one crop.
    One row per (year, planting_date) combination.
    Features from planting date, targets from future window.
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
                continue

            plant_row = df[df["date"] == plant_date]
            if plant_row.empty:
                continue

            # Extract features from planting date
            features = {}
            for col in FEATURE_COLS:
                val = plant_row[col].iloc[0] \
                    if col in plant_row.columns else np.nan
                features[col] = val if not pd.isna(val) else 0

            # Future windows for targets
            frost_window = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=FROST_CHECK_DAYS))
            ]
            full_window  = df[
                (df["date"] > plant_date) &
                (df["date"] <= plant_date +
                 pd.Timedelta(days=grow_days))
            ]

            if len(frost_window) < FROST_CHECK_DAYS * 0.5:
                continue
            if len(full_window)  < grow_days * 0.5:
                continue

            # Target 1: frost kill
            min_temp = frost_window["temp_min"].min()
            frost_kill_occurred = int(
                min_temp <= frost_kill
            )

            # Target 2: heat stress
            heat_days = (
                full_window["temp_max"] >= heat_temp
            ).sum()
            heat_occurred = int(heat_days >= min_heat_days)

            # Target 3: rainfall deficit
            monthly_rain = full_window.groupby(
                full_window["date"].dt.month
            )["precipitation"].sum()

            rain_deficit = 0
            for m, rain_mm in monthly_rain.items():
                threshold = get_critical_rain_threshold(
                    crop_key, m
                )
                if (threshold is not None and
                        rain_mm < threshold):
                    rain_deficit = 1
                    break

            row = {
                "year"        : year,
                "month"       : month,
                "day"         : day,
                "plant_date"  : plant_date,
                "crop_key"    : crop_key,
                "frost_kill"  : frost_kill_occurred,
                "heat_stress" : heat_occurred,
                "rain_deficit": rain_deficit,
                "any_risk"    : int(
                    frost_kill_occurred or
                    heat_occurred or
                    rain_deficit
                ),
            }
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------
# STEP 5 - TRAIN AND EVALUATE
# -------------------------------------------------------

def train_and_evaluate(dataset, target_col, crop_key,
                        baseline_probs=None):
    """
    Train models and evaluate with walk-forward validation.
    """
    available_features = [
        c for c in FEATURE_COLS
        if c in dataset.columns
    ]

    dataset_clean = dataset[
        available_features + [target_col, "year"]
    ].dropna()

    if len(dataset_clean) < 30:
        print(f"  Insufficient data for {target_col}")
        return None

    train_mask = dataset_clean["year"] <= TRAIN_END
    test_mask  = dataset_clean["year"] > VALIDATE_END

    X_train = dataset_clean[train_mask][available_features]
    y_train = dataset_clean[train_mask][target_col]
    X_test  = dataset_clean[test_mask][available_features]
    y_test  = dataset_clean[test_mask][target_col]

    print(f"\n  Training: {len(X_train)} samples "
          f"| Test: {len(X_test)} samples")
    print(f"  Class balance: "
          f"{y_train.sum()}/{len(y_train)} positive "
          f"({y_train.mean()*100:.0f}%)")

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print(f"  Skipping: constant target")
        return None

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        min_samples_leaf = 3,
        random_state     = 42,
        class_weight     = "balanced",
    )
    rf.fit(X_train, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators  = 200,
        max_depth     = 4,
        learning_rate = 0.05,
        random_state  = 42,
        subsample     = 0.8,
    )
    gb.fit(X_train, y_train)

    results = {}

    for name, model in [("RandomForest", rf), ("GradBoost", gb)]:
        if len(X_test) > 0 and y_test.nunique() > 1:
            y_prob   = model.predict_proba(X_test)[:, 1]
            y_pred   = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

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
            print(f"    Accuracy:    {accuracy*100:.1f}%")
            print(f"    AUC-ROC:     {auc:.3f}")
            print(f"    Brier Score: {brier:.3f}")

    # Baseline comparison
    if baseline_probs and len(X_test) > 0:
        bp = np.full(len(y_test),
                     baseline_probs.get(target_col, 0.5))
        if y_test.nunique() > 1:
            baseline_brier = brier_score_loss(y_test, bp)
            print(f"\n  Baseline Brier: {baseline_brier:.3f}")
            if "RandomForest" in results:
                diff = (baseline_brier -
                        results["RandomForest"]["brier"])
                if diff > 0.01:
                    print(f"  → RF improves by {diff:.3f} ✓")
                elif diff < -0.01:
                    print(f"  → RF worse than baseline ✗")
                else:
                    print(f"  → RF similar to baseline ≈")

    return results


# -------------------------------------------------------
# STEP 6 - FEATURE IMPORTANCE
# -------------------------------------------------------

def show_feature_importance(model, target_name, top_n=10):
    """Show which features the model found most useful."""
    available = [
        c for c in FEATURE_COLS
        if c in model.feature_names_in_
    ] if hasattr(model, "feature_names_in_") else FEATURE_COLS

    importance = pd.Series(
        model.feature_importances_,
        index=available
    ).sort_values(ascending=False)

    print(f"\n  Top {top_n} features for {target_name}:")
    for feat, imp in importance.head(top_n).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {imp:.3f}  {bar}")

    return importance


# -------------------------------------------------------
# STEP 7 - PREDICT ALL PLANTING DATES
# -------------------------------------------------------

def predict_all_dates(models, crop_key):
    """Generate risk predictions for all planting dates."""
    recent_df = df[df["year"].between(2015, 2026)].copy()
    predictions = []

    for month, day in PLANTING_CANDIDATES:
        try:
            date_label = pd.Timestamp(
                year=2024, month=month, day=day
            ).strftime("%b %d")
        except ValueError:
            continue

        date_rows = recent_df[
            (recent_df["month"] == month) &
            (recent_df["day"]   == day)
        ]

        available_features = [
            c for c in FEATURE_COLS
            if c in date_rows.columns
        ]
        date_features = date_rows[available_features].dropna()

        if date_features.empty:
            continue

        row_pred = {
            "date_label": date_label,
            "month"     : month,
            "day"       : day,
        }

        for target_name, model in models.items():
            if model is not None:
                try:
                    # Get features the model was trained on
                    if hasattr(model, "feature_names_in_"):
                        model_features = [
                            f for f in model.feature_names_in_
                            if f in date_features.columns
                        ]
                        feat_data = date_features[model_features]
                    else:
                        feat_data = date_features[
                            available_features
                        ]

                    probs = model.predict_proba(feat_data)[:, 1]
                    row_pred[f"{target_name}_prob"] = probs.mean()
                except Exception:
                    row_pred[f"{target_name}_prob"] = np.nan
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

    print(f"\nBuilding dataset...")
    dataset = build_dataset(crop_key)
    print(f"Dataset rows: {len(dataset)}")

    if dataset.empty:
        continue

    print(f"\nTarget frequencies:")
    for target in ["frost_kill", "heat_stress", "rain_deficit"]:
        freq = dataset[target].mean() * 100
        print(f"  {target:<20}: {freq:.1f}%")

    baseline_probs = {
        t: dataset[t].mean()
        for t in ["frost_kill", "heat_stress",
                  "rain_deficit", "any_risk"]
    }

    best_models = {}

    for target in ["frost_kill", "heat_stress", "rain_deficit"]:
        print(f"\n--- TARGET: {target.upper()} ---")
        results = train_and_evaluate(
            dataset, target, crop_key, baseline_probs
        )

        if results and "RandomForest" in results:
            rf_model = results["RandomForest"]["model"]
            show_feature_importance(rf_model, target, top_n=8)
            best_models[target] = rf_model
        else:
            best_models[target] = None

    print(f"\n--- PREDICTIONS FOR ALL PLANTING DATES ---")
    predictions = predict_all_dates(best_models, crop_key)
    predictions["crop"]     = crop_name
    predictions["crop_key"] = crop_key
    all_ml_predictions.append(predictions)

    if not predictions.empty:
        print(f"\n{'Date':<12} {'Frost%':>8} "
              f"{'Heat%':>8} {'Rain%':>8}")
        print("-" * 40)
        for _, row in predictions.iterrows():
            frost = row.get("frost_kill_prob",  np.nan)
            heat  = row.get("heat_stress_prob", np.nan)
            rain  = row.get("rain_deficit_prob",np.nan)

            print(f"{row['date_label']:<12} "
                  f"{frost*100:.1f}% " if not pd.isna(frost)
                  else f"{'N/A':>9} ",
                  end="")
            print(f"{heat*100:.1f}% " if not pd.isna(heat)
                  else f"{'N/A':>8} ",
                  end="")
            print(f"{rain*100:.1f}%" if not pd.isna(rain)
                  else "N/A")

# -------------------------------------------------------
# STEP 9 - SAVE
# -------------------------------------------------------

if all_ml_predictions:
    ml_combined = pd.concat(
        all_ml_predictions, ignore_index=True
    )
    output_file = os.path.join(
        OUTPUT_FOLDER, "ml_planting_predictions.csv"
    )
    ml_combined.to_csv(output_file, index=False)
    print(f"\nML predictions saved to: {output_file}")

print("\n" + "=" * 60)
print("FEATURE EXPANSION SUMMARY")
print("=" * 60)
print(f"""
New features added vs previous version:

  Soil moisture:
    soil_wet_root         Root zone wetness (0-1)
    soil_moisture_pctl    Drought percentile (0-100)
    soil_wet_surface      Surface wetness

  Soil temperature:
    soil_temp_0_10cm      Surface soil temp
    soil_temp_10_40cm     Shallow root zone temp

  Dew point:
    dew_point             Frost risk indicator
    frost_margin          Min temp minus dew point

  Water balance:
    evapotranspiration    Actual ET mm/day
    water_balance_30day   Rain minus ET (30 day)

  Accumulated GDD:
    cumulative_gdd_10     Heat units since Jan 1
    cumulative_gdd_7      Heat units since Jan 1

  Cloud cover:
    cloud_cover           Cloud percentage

Total features: {len(FEATURE_COLS)}
""")

print("\n=== MACHINE LEARNING v2 COMPLETE ===")
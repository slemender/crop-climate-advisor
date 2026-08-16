# src/risk_model.py
#
# Planting Date Risk Model v5 — Vojvodina, Serbia
#
# Calculates frost, heat, and rainfall risk for every
# candidate planting date for every crop.
#
# v5 additions:
#   - Autumn/winter planting dates (Oct, Nov, Dec)
#   - Winter survival analysis for cool-season crops
#   - Warm-season crops automatically skip autumn dates

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crop_model import CROPS, get_crop, get_frost_kill_temp
from crop_model import get_heat_stress_temp, get_monthly_water_need
from crop_model import get_critical_rain_threshold
from config import CONFIG

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE     = CONFIG["clean_data_file"]
OUTPUT_FOLDER  = "data/processed"
ANALYSIS_START = CONFIG["analysis_start"]
ANALYSIS_END   = CONFIG["analysis_end"]

FROST_CHECK_DAYS      = 21
MAX_DRY_DAYS_CRITICAL = 14
AUTUMN_MONTHS         = {10, 11, 12}
WARM_FROST_VETO       = 0.10

# Growing season days per crop
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

# For autumn-planted crops the season spans winter
# so we use a longer window
AUTUMN_GROWING_DAYS = {
    "onion"  : 210,
    "potato" : 180,
}

# Minimum heat days to count as meaningful stress
MIN_HEAT_DAYS = {
    key: 3 if crop["season_type"] == "cool" else 5
    for key, crop in CROPS.items()
}

# -------------------------------------------------------
# STEP 1 - LOAD DATA
# -------------------------------------------------------

print("=" * 60)
print("PLANTING DATE RISK MODEL v5")
print("Vojvodina, Serbia — Spring + Autumn Dates")
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

print(f"Analysis period: {ANALYSIS_START} to {ANALYSIS_END}")
print(f"Complete years:  {len(complete_years)}")
print(f"Daily rows:      {len(df):,}")

# -------------------------------------------------------
# STEP 2 - ROLLING FEATURES
# -------------------------------------------------------

print("\nBuilding rolling features...")

df = df.sort_values("date").reset_index(drop=True)

df["temp_avg_7day"]  = df["temp_avg"].rolling(7,  min_periods=1).mean()
df["temp_max_7day"]  = df["temp_max"].rolling(7,  min_periods=1).mean()
df["temp_avg_30day"] = df["temp_avg"].rolling(30, min_periods=1).mean()
df["temp_max_30day"] = df["temp_max"].rolling(30, min_periods=1).mean()
df["rain_7day"]      = df["precipitation"].rolling(7,  min_periods=1).sum()
df["rain_14day"]     = df["precipitation"].rolling(14, min_periods=1).sum()
df["rain_30day"]     = df["precipitation"].rolling(30, min_periods=1).sum()

def count_consecutive_dry(series, threshold=1.0):
    result  = []
    counter = 0
    for val in series:
        if val < threshold:
            counter += 1
        else:
            counter = 0
        result.append(counter)
    return result

df["consecutive_dry_days"] = count_consecutive_dry(
    df["precipitation"].values
)

monthly_avg_temp   = df.groupby("month")["temp_avg"].transform("mean")
df["temp_anomaly"] = df["temp_avg"] - monthly_avg_temp

monthly_avg_rain   = df.groupby("month")["precipitation"].transform("mean")
df["rain_anomaly"] = df["precipitation"] - monthly_avg_rain

print("Features built.")

# -------------------------------------------------------
# STEP 3 - HELPER: GET WEATHER WINDOW
# -------------------------------------------------------

def get_window(df, planting_month, planting_day, n_days):
    """Extract n_days of weather starting from planting date."""
    windows = []
    for year in complete_years:
        try:
            plant_date = pd.Timestamp(
                year=year,
                month=planting_month,
                day=planting_day
            )
        except ValueError:
            continue

        future = df[df["date"] >= plant_date].head(n_days)

        if len(future) >= int(n_days * 0.8):
            windows.append({"year": year, "data": future})

    return windows


# -------------------------------------------------------
# STEP 4 - WINTER SURVIVAL (AUTUMN PLANTING)
# -------------------------------------------------------

def calculate_winter_survival(df, planting_month,
                               planting_day, crop_key):
    """
    For autumn-planted crops, calculate the probability
    of surviving the winter through to spring.

    We check the minimum temperature from planting date
    through March 1 of the following year against a
    winter survival threshold.

    Onion bulbs in soil can survive to about -12°C.
    Potato tubers are damaged below about -3°C in soil.
    """
    crop = get_crop(crop_key)

    # Winter kill temperature depends on crop
    if crop_key == "onion":
        # Established onion bulbs are very hardy
        winter_kill_temp = -12.0
    else:
        # Use foliage kill temp for other crops
        winter_kill_temp = crop["frost"]["foliage_kill_c"]

    results = []

    for year in complete_years:
        try:
            plant_date = pd.Timestamp(
                year=year,
                month=planting_month,
                day=planting_day
            )
            winter_end = pd.Timestamp(
                year=year + 1, month=3, day=1
            )
        except ValueError:
            continue

        winter_data = df[
            (df["date"] >= plant_date) &
            (df["date"] <= winter_end)
        ]

        if len(winter_data) < 30:
            continue

        min_temp = winter_data["temp_min"].min()
        survived = int(min_temp > winter_kill_temp)

        results.append({
            "year"       : year,
            "min_temp"   : min_temp,
            "survived"   : survived,
        })

    if not results:
        return None

    n              = len(results)
    survived_count = sum(r["survived"] for r in results)
    min_temps      = [r["min_temp"] for r in results]

    return {
        "winter_survival_prob": survived_count / n,
        "winter_kill_prob"    : 1 - (survived_count / n),
        "avg_winter_min_temp" : round(np.mean(min_temps), 1),
        "worst_winter_temp"   : round(min(min_temps), 1),
        "n_years"             : n,
    }


# -------------------------------------------------------
# STEP 5 - FROST RISK (STANDARD 21-DAY)
# -------------------------------------------------------

def calculate_frost_risk(df, planting_month,
                          planting_day, crop_key):
    """Frost risk in the 21 days after planting."""
    crop              = get_crop(crop_key)
    frost_kill_temp   = crop["frost"]["foliage_kill_c"]
    frost_damage_temp = crop["frost"]["foliage_damage_c"]

    windows = get_window(
        df, planting_month, planting_day, FROST_CHECK_DAYS
    )
    if not windows:
        return None

    n          = len(windows)
    kill_count = 0
    dmg_count  = 0
    worst_temp = 99.0
    kill_years = []

    for w in windows:
        data     = w["data"]
        year     = w["year"]
        min_temp = data["temp_min"].min()

        if min_temp < worst_temp:
            worst_temp = min_temp
        if min_temp <= frost_damage_temp:
            dmg_count += 1
        if min_temp <= frost_kill_temp:
            kill_count += 1
            kill_years.append(year)

    return {
        "prob_damage_frost": dmg_count  / n,
        "prob_kill_frost"  : kill_count / n,
        "worst_temp_c"     : worst_temp,
        "kill_years"       : kill_years,
        "n_years"          : n,
    }


# -------------------------------------------------------
# STEP 6 - HEAT RISK
# -------------------------------------------------------

def calculate_heat_risk(df, planting_month,
                         planting_day, crop_key,
                         is_autumn=False):
    """
    Heat stress risk during the growing season.
    For autumn planting uses the longer autumn growing window
    which covers the following spring and summer.
    """
    heat_temp = get_heat_stress_temp(crop_key)
    mod_temp  = heat_temp - 5
    min_days  = MIN_HEAT_DAYS[crop_key]

    grow_days = (
        AUTUMN_GROWING_DAYS.get(crop_key, GROWING_SEASON_DAYS[crop_key])
        if is_autumn
        else GROWING_SEASON_DAYS[crop_key]
    )

    windows = get_window(
        df, planting_month, planting_day, grow_days
    )
    if not windows:
        return None

    n             = len(windows)
    severe_count  = 0
    mod_count     = 0
    heat_day_list = []
    worst_temp    = -99.0
    severe_years  = []

    for w in windows:
        data      = w["data"]
        year      = w["year"]
        max_temp  = data["temp_max"].max()
        heat_days = (data["temp_max"] >= heat_temp).sum()
        mod_days  = (data["temp_max"] >= mod_temp).sum()

        if max_temp > worst_temp:
            worst_temp = max_temp

        heat_day_list.append(int(heat_days))

        if mod_days >= max(1, min_days - 2):
            mod_count += 1
        if heat_days >= min_days:
            severe_count += 1
            severe_years.append(year)

    return {
        "prob_moderate_heat": mod_count    / n,
        "prob_severe_heat"  : severe_count / n,
        "avg_heat_days"     : round(np.mean(heat_day_list), 1),
        "max_heat_days"     : int(np.max(heat_day_list)),
        "worst_temp_c"      : worst_temp,
        "severe_years"      : severe_years,
        "n_years"           : n,
    }


# -------------------------------------------------------
# STEP 7 - RAINFALL RISK
# -------------------------------------------------------

def calculate_rainfall_risk(df, planting_month,
                              planting_day, crop_key,
                              is_autumn=False):
    """
    Rainfall adequacy over the full growing season.
    For autumn planting uses the longer autumn window so
    that spring drought risk is captured.
    """
    grow_days = (
        AUTUMN_GROWING_DAYS.get(crop_key, GROWING_SEASON_DAYS[crop_key])
        if is_autumn
        else GROWING_SEASON_DAYS[crop_key]
    )

    windows = get_window(
        df, planting_month, planting_day, grow_days
    )
    if not windows:
        return None

    n                   = len(windows)
    deficit_count       = 0
    drought_spell_count = 0
    total_rains         = []
    deficit_years       = []

    for w in windows:
        data = w["data"]
        year = w["year"]

        total_rain = data["precipitation"].sum()
        total_rains.append(total_rain)

        monthly_rain = data.groupby(
            data["date"].dt.month
        )["precipitation"].sum()

        year_has_deficit = False
        for month, rain_mm in monthly_rain.items():
            threshold = get_critical_rain_threshold(
                crop_key, month
            )
            if threshold is not None and rain_mm < threshold:
                year_has_deficit = True

        if year_has_deficit:
            deficit_count += 1
            deficit_years.append(year)

        max_dry = data["consecutive_dry_days"].max()
        if max_dry >= MAX_DRY_DAYS_CRITICAL:
            drought_spell_count += 1

    return {
        "prob_rainfall_deficit" : deficit_count       / n,
        "prob_drought_spell"    : drought_spell_count / n,
        "avg_total_rain_mm"     : round(np.mean(total_rains), 1),
        "min_total_rain_mm"     : round(np.min(total_rains),  1),
        "max_total_rain_mm"     : round(np.max(total_rains),  1),
        "deficit_years"         : deficit_years,
        "n_years"               : n,
    }


# -------------------------------------------------------
# STEP 8 - COMBINED RISK SCORE
# -------------------------------------------------------

def calculate_combined_risk(frost_r, heat_r, rain_r,
                              crop_key, winter_r=None):
    """
    Combine frost, heat, and rainfall into a single
    risk score 0 (lowest) to 1 (highest).

    For autumn planting, winter_r replaces frost_r
    as the primary cold-weather risk measure.
    """
    crop = get_crop(crop_key)

    # Use winter survival if provided (autumn planting)
    if winter_r is not None:
        frost_kill_prob = winter_r["winter_kill_prob"]
    else:
        frost_kill_prob = frost_r["prob_kill_frost"] \
            if frost_r else 0

    heat_score = heat_r["prob_severe_heat"]      if heat_r else 0
    rain_score = rain_r["prob_rainfall_deficit"] if rain_r else 0

    if crop["season_type"] == "cool":
        weights = {
            "frost"    : 0.35,
            "heat"     : 0.30,
            "rainfall" : 0.35,
        }
    else:
        weights = {
            "frost"    : 0.30,
            "heat"     : 0.28,
            "rainfall" : 0.42,
        }

    combined = (
        frost_kill_prob * weights["frost"]    +
        heat_score      * weights["heat"]     +
        rain_score      * weights["rainfall"]
    )

    # Extra penalty when heat and drought both occur
    combined += heat_score * rain_score * 0.15

    # Frost veto for warm-season crops in spring
    # (not applied to autumn planting)
    if (winter_r is None and
            crop["season_type"] == "warm" and
            frost_kill_prob > WARM_FROST_VETO):
        combined += (frost_kill_prob - WARM_FROST_VETO) * 0.50

    return {
        "frost_score"    : frost_kill_prob,
        "heat_score"     : heat_score,
        "rain_score"     : rain_score,
        "combined_score" : min(1.0, max(0.0, combined)),
    }


# -------------------------------------------------------
# STEP 9 - LABELS
# -------------------------------------------------------

def suitability_label(score):
    if score < 0.15:   return "Excellent"
    elif score < 0.25: return "Good"
    elif score < 0.40: return "Moderate"
    elif score < 0.55: return "Poor"
    else:              return "Very Poor"


# -------------------------------------------------------
# STEP 10 - CANDIDATE PLANTING DATES
# -------------------------------------------------------

PLANTING_CANDIDATES = [
    # Autumn dates (cool-season crops only)
    (10,  1),
    (10, 15),
    (11,  1),
    (11, 15),
    (12,  1),

    # Spring dates
    (2,  15),
    (3,   1),
    (3,  10),
    (3,  20),
    (4,   1),
    (4,  10),
    (4,  20),
    (5,   1),
    (5,  10),
    (5,  20),
    (6,   1),
]


# -------------------------------------------------------
# STEP 11 - RUN ANALYSIS
# -------------------------------------------------------

all_results = []

print("\n" + "=" * 60)
print("CALCULATING RISK FOR ALL CROPS AND PLANTING DATES")
print("=" * 60)

for crop_key in CROPS.keys():
    crop      = get_crop(crop_key)
    crop_name = crop["name"]

    print(f"\n{'='*60}")
    print(f"CROP: {crop_name} ({crop['season_type']}-season)")
    print(f"Frost kill: {get_frost_kill_temp(crop_key)}°C  |  "
          f"Heat stress: {get_heat_stress_temp(crop_key)}°C")
    print(f"{'='*60}")

    crop_results = []

    for month, day in PLANTING_CANDIDATES:

        try:
            date_label = pd.Timestamp(
                year=2024, month=month, day=day
            ).strftime("%b %d")
        except ValueError:
            continue

        is_autumn = month in AUTUMN_MONTHS

        # Skip autumn dates for warm-season crops
        if is_autumn and crop["season_type"] == "warm":
            continue

        # Calculate risk components
        frost_r  = calculate_frost_risk(
            df, month, day, crop_key
        )
        heat_r   = calculate_heat_risk(
            df, month, day, crop_key, is_autumn
        )
        rain_r   = calculate_rainfall_risk(
            df, month, day, crop_key, is_autumn
        )

        # Winter survival for autumn dates
        winter_r = None
        if is_autumn:
            winter_r = calculate_winter_survival(
                df, month, day, crop_key
            )

        if not all([frost_r, heat_r, rain_r]):
            continue

        combined = calculate_combined_risk(
            frost_r, heat_r, rain_r, crop_key, winter_r
        )

        # Display
        frost_display = (
            f"W.kill {combined['frost_score']*100:.1f}%"
            if is_autumn
            else f"Frost  {combined['frost_score']*100:.1f}%"
        )

        print(f"  {date_label}  {frost_display}  "
              f"Heat {heat_r['prob_severe_heat']*100:.1f}%  "
              f"Rain {rain_r['prob_rainfall_deficit']*100:.1f}%  "
              f"→ {combined['combined_score']*100:.1f}% "
              f"{suitability_label(combined['combined_score'])}")

        if is_autumn and winter_r:
            surv = winter_r["winter_survival_prob"] * 100
            avg  = winter_r["avg_winter_min_temp"]
            worst = winter_r["worst_winter_temp"]
            print(f"          └ Winter survival: {surv:.0f}%  "
                  f"Avg min: {avg}°C  Worst: {worst}°C")

        # Store result
        crop_results.append({
            "crop"                 : crop_name,
            "crop_key"             : crop_key,
            "month"                : month,
            "day"                  : day,
            "date_label"           : date_label,
            "is_autumn_planting"   : is_autumn,

            # Frost
            "frost_damage_prob"    : frost_r["prob_damage_frost"],
            "frost_kill_prob"      : combined["frost_score"],
            "frost_worst_temp_c"   : frost_r["worst_temp_c"],

            # Winter survival (autumn only)
            "winter_survival_prob" : winter_r[
                "winter_survival_prob"
            ] if winter_r else np.nan,
            "winter_avg_min_temp"  : winter_r[
                "avg_winter_min_temp"
            ] if winter_r else np.nan,
            "winter_worst_temp"    : winter_r[
                "worst_winter_temp"
            ] if winter_r else np.nan,

            # Heat
            "heat_moderate_prob"   : heat_r["prob_moderate_heat"],
            "heat_severe_prob"     : heat_r["prob_severe_heat"],
            "heat_avg_days"        : heat_r["avg_heat_days"],
            "heat_max_days"        : heat_r["max_heat_days"],
            "heat_worst_temp_c"    : heat_r["worst_temp_c"],

            # Rain
            "rain_deficit_prob"    : rain_r["prob_rainfall_deficit"],
            "rain_drought_prob"    : rain_r["prob_drought_spell"],
            "rain_avg_total_mm"    : rain_r["avg_total_rain_mm"],
            "rain_min_total_mm"    : rain_r["min_total_rain_mm"],

            # Combined
            "combined_score"       : combined["combined_score"],
            "frost_score"          : combined["frost_score"],
            "heat_score"           : combined["heat_score"],
            "rain_score"           : combined["rain_score"],
            "suitability"          : suitability_label(
                combined["combined_score"]
            ),
            "n_years"              : frost_r["n_years"],
        })

    all_results.extend(crop_results)

    # Summary per crop
    if crop_results:
        spring = [r for r in crop_results
                  if not r["is_autumn_planting"]]
        autumn = [r for r in crop_results
                  if r["is_autumn_planting"]]

        if spring:
            best_s = min(spring, key=lambda x: x["combined_score"])
            print(f"\n  Best spring: {best_s['date_label']} "
                  f"({best_s['combined_score']*100:.1f}% "
                  f"— {best_s['suitability']})")

        if autumn:
            best_a = min(autumn, key=lambda x: x["combined_score"])
            surv   = best_a["winter_survival_prob"]
            surv_s = f"{surv*100:.0f}% survive" \
                if not pd.isna(surv) else ""
            print(f"  Best autumn: {best_a['date_label']} "
                  f"({best_a['combined_score']*100:.1f}% "
                  f"— {best_a['suitability']})  {surv_s}")


# -------------------------------------------------------
# STEP 12 - SUMMARY
# -------------------------------------------------------

print("\n\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(all_results)

for crop_key in CROPS.keys():
    cd        = results_df[results_df["crop_key"] == crop_key]
    crop_name = CROPS[crop_key]["name"]
    if cd.empty:
        continue

    spring = cd[~cd["is_autumn_planting"]]
    autumn = cd[cd["is_autumn_planting"]]

    print(f"\n{crop_name}")

    if not spring.empty:
        best = spring.nsmallest(1, "combined_score").iloc[0]
        print(f"  Spring best:  {best['date_label']}  "
              f"{best['combined_score']*100:.1f}%  "
              f"{best['suitability']}")

    if not autumn.empty:
        best = autumn.nsmallest(1, "combined_score").iloc[0]
        surv = best["winter_survival_prob"]
        surv_s = f"  ({surv*100:.0f}% winter survival)" \
            if not pd.isna(surv) else ""
        print(f"  Autumn best:  {best['date_label']}  "
              f"{best['combined_score']*100:.1f}%  "
              f"{best['suitability']}{surv_s}")
    else:
        print(f"  Autumn:       Not suitable (warm-season crop)")


# -------------------------------------------------------
# STEP 13 - SAVE
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_file = os.path.join(
    OUTPUT_FOLDER, "planting_date_risk_scores.csv"
)
results_df.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print(f"Records: {len(results_df)}")
print("\n=== RISK MODEL v5 COMPLETE ===")
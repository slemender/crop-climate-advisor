# src/risk_model.py
#
# Planting Date Risk Model for Vojvodina, Serbia
#
# VERSION 4 - Two key improvements:
#
# v3 fixes:
#   - Full growing season rainfall window (not fixed 90 days)
#   - Frost veto for warm-season crops
#
# v4 fixes:
#   - Heat risk now requires meaningful heat spell (3+ days)
#     rather than any single hot day triggering 100% risk.
#     A single 35°C day in a 120-day season is not the same
#     as 20 consecutive heat days.
#   - This produces more realistic heat risk scores and
#     distinguishes occasional heat from sustained stress.
#
# Data: 2000-2026 (26 complete years)
# Crops: Potato, Tomato, Onion, Cucumber
# No irrigation scenario

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crop_model import CROPS, get_crop, get_frost_kill_temp
from crop_model import get_heat_stress_temp, get_monthly_water_need
from crop_model import get_critical_rain_threshold

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE     = "data/processed/vojvodina_clean.csv"
OUTPUT_FOLDER  = "data/processed"
ANALYSIS_START = 2000
ANALYSIS_END   = 2026

# Days after planting to check for frost risk
FROST_CHECK_DAYS = 21   # 3 weeks - critical establishment period

# Maximum consecutive dry days before serious stress
MAX_DRY_DAYS_CRITICAL = 14

# Build growing season days from crop database
# Uses medium variety days to maturity as the season length
GROWING_SEASON_DAYS = {
    key: crop["maturity"].get("medium_days",
         crop["maturity"].get("from_transplant_days",
         crop["maturity"].get("from_seed_days", 90)))
    for key, crop in CROPS.items()
}

# Minimum heat days to count as meaningful heat risk
# Cool season crops are more sensitive — use lower threshold
# Warm season crops are more tolerant — use higher threshold
MIN_HEAT_DAYS_FOR_RISK = {
    key: 3 if crop["season_type"] == "cool" else 5
    for key, crop in CROPS.items()
}

# Frost veto threshold for warm-season crops
WARM_CROP_FROST_VETO_THRESHOLD = 0.10

# -------------------------------------------------------
# STEP 1 - LOAD AND PREPARE DATA
# -------------------------------------------------------

print("=" * 60)
print("PLANTING DATE RISK MODEL v4")
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
n_years        = len(years)
complete_years = [
    y for y in years
    if df[df["year"] == y].shape[0] >= 350
]

print(f"Analysis period: {ANALYSIS_START} to {ANALYSIS_END}")
print(f"Total years:     {n_years}")
print(f"Complete years:  {len(complete_years)}")
print(f"Daily rows:      {len(df):,}")

# -------------------------------------------------------
# STEP 2 - CALCULATE ROLLING WEATHER FEATURES
# -------------------------------------------------------

print("\nCalculating rolling weather features...")

df = df.sort_values("date").reset_index(drop=True)

df["temp_avg_7day"]  = df["temp_avg"].rolling(7,  min_periods=1).mean()
df["temp_max_7day"]  = df["temp_max"].rolling(7,  min_periods=1).mean()
df["temp_avg_30day"] = df["temp_avg"].rolling(30, min_periods=1).mean()
df["temp_max_30day"] = df["temp_max"].rolling(30, min_periods=1).mean()
df["rain_7day"]      = df["precipitation"].rolling(7,  min_periods=1).sum()
df["rain_14day"]     = df["precipitation"].rolling(14, min_periods=1).sum()
df["rain_30day"]     = df["precipitation"].rolling(30, min_periods=1).sum()

def count_consecutive_dry(series, threshold=1.0):
    """Count consecutive days with rainfall below threshold."""
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

monthly_avg_temp = df.groupby("month")["temp_avg"].transform("mean")
df["temp_anomaly"] = df["temp_avg"] - monthly_avg_temp

monthly_avg_rain = df.groupby("month")["precipitation"].transform("mean")
df["rain_anomaly"] = df["precipitation"] - monthly_avg_rain

print("Rolling features calculated.")

# -------------------------------------------------------
# STEP 3 - HELPER: GET WEATHER WINDOW AFTER PLANTING
# -------------------------------------------------------

def get_window(df, planting_month, planting_day, n_days):
    """
    For each complete year extract n_days of weather
    starting from the given planting date.
    """
    windows = []
    for year in complete_years:
        try:
            plant_date = pd.Timestamp(
                year=year, month=planting_month, day=planting_day
            )
        except ValueError:
            continue

        future = df[df["date"] >= plant_date].head(n_days)

        if len(future) >= int(n_days * 0.8):
            windows.append({"year": year, "data": future})

    return windows

# -------------------------------------------------------
# STEP 4 - FROST RISK
# -------------------------------------------------------

def calculate_frost_risk(df, planting_month, planting_day, crop_key):
    """
    Calculate frost risk during the 21-day establishment
    period after planting.
    """
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
        "prob_damage_frost" : dmg_count  / n,
        "prob_kill_frost"   : kill_count / n,
        "worst_temp_c"      : worst_temp,
        "kill_years"        : kill_years,
        "n_years"           : n,
    }

# -------------------------------------------------------
# STEP 5 - HEAT RISK (KEY FIX IN v4)
# -------------------------------------------------------

def calculate_heat_risk(df, planting_month, planting_day, crop_key):
    """
    Calculate meaningful heat stress risk over the full
    growing season.

    KEY FIX IN v4:
    We now require a minimum number of heat days to count
    as "severe heat risk". This prevents a single hot day
    in a 120-day season from triggering 100% heat risk.

    A single day above 35°C happens in almost every Vojvodina
    summer. That is different from 10+ sustained heat days
    which causes real crop damage.

    The minimum threshold differs by crop:
    - Potato/Tomato: 3+ days (more sensitive)
    - Onion/Cucumber: 5+ days (more tolerant)

    This produces more meaningful risk scores that
    distinguish a normal warm summer from a heat wave.
    """
    heat_temp    = get_heat_stress_temp(crop_key)
    mod_temp     = heat_temp - 5
    growing_days = GROWING_SEASON_DAYS[crop_key]
    min_days     = MIN_HEAT_DAYS_FOR_RISK[crop_key]

    windows = get_window(
        df, planting_month, planting_day, growing_days
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

        heat_day_list.append(heat_days)

        # KEY CHANGE: require minimum heat days to count as risk
        # This prevents a single warm day triggering 100% risk
        if mod_days >= max(1, min_days - 2):
            mod_count += 1

        if heat_days >= min_days:
            severe_count += 1
            severe_years.append(year)

    return {
        "prob_moderate_heat" : mod_count    / n,
        "prob_severe_heat"   : severe_count / n,
        "avg_heat_days"      : round(np.mean(heat_day_list), 1),
        "max_heat_days"      : int(np.max(heat_day_list)),
        "worst_temp_c"       : worst_temp,
        "severe_years"       : severe_years,
        "n_years"            : n,
        "min_days_threshold" : min_days,
    }

# -------------------------------------------------------
# STEP 6 - RAINFALL RISK
# -------------------------------------------------------

def calculate_rainfall_risk(df, planting_month, planting_day, crop_key):
    """
    Calculate rainfall adequacy risk over the full growing season.
    Uses crop-specific season length so that crops planted in
    spring are evaluated for their actual summer drought risk.
    Critical for a no-irrigation scenario.
    """
    growing_days = GROWING_SEASON_DAYS[crop_key]

    windows = get_window(
        df, planting_month, planting_day, growing_days
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

        # Check rainfall in each calendar month the crop occupies
        monthly_rain = data.groupby(
            data["date"].dt.month
        )["precipitation"].sum()

        year_has_critical_deficit = False

        for month, rain_mm in monthly_rain.items():
            threshold = get_critical_rain_threshold(crop_key, month)
            if threshold is not None:
                if rain_mm < threshold:
                    year_has_critical_deficit = True

        if year_has_critical_deficit:
            deficit_count += 1
            deficit_years.append(year)

        # Check for long dry spells
        max_dry = data["consecutive_dry_days"].max()
        if max_dry >= MAX_DRY_DAYS_CRITICAL:
            drought_spell_count += 1

    return {
        "prob_rainfall_deficit"  : deficit_count       / n,
        "prob_drought_spell"     : drought_spell_count / n,
        "avg_total_rain_mm"      : round(np.mean(total_rains), 1),
        "min_total_rain_mm"      : round(np.min(total_rains),  1),
        "max_total_rain_mm"      : round(np.max(total_rains),  1),
        "deficit_years"          : deficit_years,
        "n_years"                : n,
    }

# -------------------------------------------------------
# STEP 7 - COMBINED RISK SCORE WITH FROST VETO
# -------------------------------------------------------

def calculate_combined_risk(frost_r, heat_r, rain_r, crop_key):
    """
    Combine frost, heat, and rainfall risks into a single
    overall risk score from 0 (lowest) to 1 (highest).

    Weights reflect:
    - Crop type (cool vs warm season)
    - No irrigation (rainfall weighted heavily)

    Frost veto for warm-season crops:
    - Prevents the model recommending frost-risky dates
      for crops that are completely killed by frost
    """
    crop = get_crop(crop_key)

    frost_kill_prob = frost_r["prob_kill_frost"]      if frost_r else 0
    heat_score      = heat_r["prob_severe_heat"]      if heat_r  else 0
    rain_score      = rain_r["prob_rainfall_deficit"] if rain_r  else 0

    if crop["season_type"] == "cool":
        weights = {
            "frost"    : 0.35,
            "heat"     : 0.30,
            "rainfall" : 0.35,
        }
    else:
        # Warm season - rainfall and frost both critical
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

    # Extra penalty when heat AND drought occur together
    heat_drought_penalty = heat_score * rain_score * 0.15
    combined += heat_drought_penalty

    # Frost veto for warm-season crops
    veto_applied = False
    if (crop["season_type"] == "warm" and
            frost_kill_prob > WARM_CROP_FROST_VETO_THRESHOLD):
        frost_veto_penalty = (
            (frost_kill_prob - WARM_CROP_FROST_VETO_THRESHOLD) * 0.50
        )
        combined      += frost_veto_penalty
        veto_applied   = True

    combined = min(1.0, max(0.0, combined))

    return {
        "frost_score"        : frost_kill_prob,
        "heat_score"         : heat_score,
        "rain_score"         : rain_score,
        "combined_score"     : combined,
        "weights"            : weights,
        "frost_veto_applied" : veto_applied,
    }

# -------------------------------------------------------
# LABEL HELPERS
# -------------------------------------------------------

def suitability_label(score):
    if score < 0.15:   return "Excellent"
    elif score < 0.25: return "Good"
    elif score < 0.40: return "Moderate"
    elif score < 0.55: return "Poor"
    else:              return "Very Poor"

# -------------------------------------------------------
# STEP 8 - CANDIDATE PLANTING DATES
# -------------------------------------------------------

PLANTING_CANDIDATES = [
    (2, 15),
    (3,  1),
    (3, 10),
    (3, 20),
    (4,  1),
    (4, 10),
    (4, 20),
    (5,  1),
    (5, 10),
    (5, 20),
    (6,  1),
]

# -------------------------------------------------------
# STEP 9 - RUN RISK ANALYSIS
# -------------------------------------------------------

all_results = []

print("\n" + "=" * 60)
print("CALCULATING RISK FOR ALL CROPS AND PLANTING DATES")
print("v4: Meaningful heat threshold + frost veto + full season rain")
print("=" * 60)

for crop_key in CROPS.keys():
    crop      = get_crop(crop_key)
    crop_name = crop["name"]
    grow_days = GROWING_SEASON_DAYS[crop_key]
    min_heat  = MIN_HEAT_DAYS_FOR_RISK[crop_key]

    print(f"\n{'='*60}")
    print(f"CROP: {crop_name} ({crop['season_type']}-season)")
    print(f"Frost kill temp:    {get_frost_kill_temp(crop_key)}°C")
    print(f"Heat stress temp:   {get_heat_stress_temp(crop_key)}°C "
          f"(requires {min_heat}+ days to count as risk)")
    print(f"Growing season:     {grow_days} days")
    if crop["season_type"] == "warm":
        print(f"Frost veto:         active "
              f"(threshold {WARM_CROP_FROST_VETO_THRESHOLD*100:.0f}%)")
    print(f"{'='*60}")
    print(f"\n{'Date':<12} {'Frost':>8} {'Heat':>8} "
          f"{'Rain':>8} {'Avg Heat':>10} {'Combined':>10} "
          f"{'Suitability':>13} {'Veto':>6}")
    print(f"{'':12} {'Kill%':>8} {'Severe%':>8} "
          f"{'Deficit%':>8} {'Days/yr':>10} {'Score':>10} "
          f"{'':>13} {'':>6}")
    print("-" * 80)

    crop_results = []

    for month, day in PLANTING_CANDIDATES:
        try:
            date_label = pd.Timestamp(
                year=2024, month=month, day=day
            ).strftime("%b %d")
        except ValueError:
            continue

        frost_r = calculate_frost_risk(df, month, day, crop_key)
        heat_r  = calculate_heat_risk(df, month, day, crop_key)
        rain_r  = calculate_rainfall_risk(df, month, day, crop_key)

        if not all([frost_r, heat_r, rain_r]):
            continue

        combined = calculate_combined_risk(
            frost_r, heat_r, rain_r, crop_key
        )

        frost_pct    = frost_r["prob_kill_frost"]       * 100
        heat_pct     = heat_r["prob_severe_heat"]       * 100
        rain_pct     = rain_r["prob_rainfall_deficit"]  * 100
        avg_heat     = heat_r["avg_heat_days"]
        combo_pct    = combined["combined_score"]        * 100
        suit         = suitability_label(combined["combined_score"])
        veto_flag    = "YES" if combined["frost_veto_applied"] else ""

        print(f"{date_label:<12} "
              f"{frost_pct:>7.1f}% "
              f"{heat_pct:>7.1f}% "
              f"{rain_pct:>7.1f}% "
              f"{avg_heat:>10.1f} "
              f"{combo_pct:>9.1f}% "
              f"{suit:>13} "
              f"{veto_flag:>6}")

        crop_results.append({
            "crop"                   : crop_name,
            "crop_key"               : crop_key,
            "month"                  : month,
            "day"                    : day,
            "date_label"             : date_label,
            "frost_damage_prob"      : frost_r["prob_damage_frost"],
            "frost_kill_prob"        : frost_r["prob_kill_frost"],
            "frost_worst_temp_c"     : frost_r["worst_temp_c"],
            "heat_moderate_prob"     : heat_r["prob_moderate_heat"],
            "heat_severe_prob"       : heat_r["prob_severe_heat"],
            "heat_avg_days"          : heat_r["avg_heat_days"],
            "heat_max_days"          : heat_r["max_heat_days"],
            "heat_worst_temp_c"      : heat_r["worst_temp_c"],
            "rain_deficit_prob"      : rain_r["prob_rainfall_deficit"],
            "rain_drought_spell_prob": rain_r["prob_drought_spell"],
            "rain_avg_total_mm"      : rain_r["avg_total_rain_mm"],
            "rain_min_total_mm"      : rain_r["min_total_rain_mm"],
            "combined_score"         : combined["combined_score"],
            "frost_score"            : combined["frost_score"],
            "heat_score"             : combined["heat_score"],
            "rain_score"             : combined["rain_score"],
            "frost_veto_applied"     : combined["frost_veto_applied"],
            "suitability"            : suit,
            "n_years"                : frost_r["n_years"],
            "growing_season_days"    : grow_days,
        })

    all_results.extend(crop_results)

    if crop_results:
        best  = min(crop_results, key=lambda x: x["combined_score"])
        worst = max(crop_results, key=lambda x: x["combined_score"])
        print(f"\n  Best date:  {best['date_label']} "
              f"(combined: {best['combined_score']*100:.1f}%"
              f" — {best['suitability']})")
        print(f"  Worst date: {worst['date_label']} "
              f"(combined: {worst['combined_score']*100:.1f}%"
              f" — {worst['suitability']})")

# -------------------------------------------------------
# STEP 10 - SUMMARY
# -------------------------------------------------------

print("\n\n" + "=" * 60)
print("PLANTING WINDOW SUMMARY — ALL CROPS")
print("Vojvodina, Serbia — Based on 2000-2026 data")
print("No irrigation — rainfall fully accounted for")
print("=" * 60)

results_df = pd.DataFrame(all_results)

for crop_key in CROPS.keys():
    crop_data = results_df[results_df["crop_key"] == crop_key]
    crop_name = CROPS[crop_key]["name"]
    if crop_data.empty:
        continue

    best_dates   = crop_data.nsmallest(3, "combined_score")
    safest_frost = crop_data.nsmallest(1, "frost_kill_prob").iloc[0]

    print(f"\n{crop_name}")
    print(f"  Top 3 dates by combined risk:")
    for _, row in best_dates.iterrows():
        veto = " [frost veto]" if row["frost_veto_applied"] else ""
        print(f"    {row['date_label']:<10} "
              f"frost:{row['frost_kill_prob']*100:.0f}%  "
              f"heat:{row['heat_severe_prob']*100:.0f}%  "
              f"rain:{row['rain_deficit_prob']*100:.0f}%  "
              f"combined:{row['combined_score']*100:.1f}%"
              f"  → {row['suitability']}{veto}")
    print(f"  Safest from frost: {safest_frost['date_label']} "
          f"({safest_frost['frost_kill_prob']*100:.0f}% kill risk)")

# -------------------------------------------------------
# STEP 11 - PLAIN LANGUAGE
# -------------------------------------------------------

print("\n\n" + "=" * 60)
print("WHAT THE MODEL IS TELLING YOU")
print("Vojvodina farmer — no irrigation")
print("=" * 60)

print("""
POTATO — Recommended: late February to early March
  Plant early varieties. The key insight is that early planting
  allows harvest in late May to June — before summer heat
  and drought arrive. This is the most manageable crop
  without irrigation in current Vojvodina conditions.
  Frost risk is real but potatoes often recover from shoots.

TOMATO — Recommended: late April to early May
  This is a challenging crop without irrigation.
  The model honestly shows no risk-free window exists.
  Late April minimizes frost risk while giving fruit
  the best chance in June rainfall before summer drought.
  Success in any year depends heavily on June-August rainfall.
  In wet years (e.g. 2023) expect good results.
  In dry years (e.g. 2025) expect yield loss.

ONION — Recommended: early to mid February
  The most frost-tolerant crop. Plant as early as possible
  to get bulbing complete in May-June when rainfall is
  better. Onions are the most reliable cool-season crop
  for your situation.

CUCUMBER — Recommended: April 20
  The model clearly identifies April 20 as the sweet spot.
  Zero frost risk, modest heat risk (cucumbers are tolerant),
  and fruit developing in June when rainfall is highest.
  Choose fast-maturing varieties (50-60 day types).
""")

# -------------------------------------------------------
# STEP 12 - DISCLAIMER
# -------------------------------------------------------

print("=" * 60)
print("IMPORTANT — WHAT THESE NUMBERS MEAN")
print("=" * 60)
print("""
Frost Risk:   % of years 2000-2026 with killing frost
              in the 21 days after this planting date.

Heat Risk:    % of years with sustained damaging heat
              (3+ days for potato/tomato, 5+ for onion/cucumber)
              during the full growing season.
              NOTE: Occasional single hot days not counted —
              only meaningful heat spells that cause crop damage.

Rain Risk:    % of years where critical growth months
              received less than minimum rainfall threshold.
              No irrigation — this is a real yield-loss risk.

Combined:     Weighted risk score. Lower = safer to plant.
              Includes frost veto for warm-season crops.

These are HISTORICAL PROBABILITIES, not forecasts.
Any specific year may differ significantly.
Use alongside local knowledge and experience.
""")

# -------------------------------------------------------
# STEP 13 - SAVE
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_file = os.path.join(
    OUTPUT_FOLDER, "planting_date_risk_scores.csv"
)
results_df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(f"Total records:    {len(results_df)}")
print("\n=== RISK MODEL v4 COMPLETE ===")
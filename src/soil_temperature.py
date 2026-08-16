# src/soil_temperature.py
#
# Download and analyze soil temperature data from NASA POWER
# for Vojvodina, Serbia.
#
# We use the same NASA POWER API we already use for air
# temperature and precipitation — no new accounts or
# libraries required.
#
# NASA POWER soil temperature variables:
#   TSOIL1 = 0-10 cm depth   (°C)
#   TSOIL2 = 10-40 cm depth  (°C)
#   TSOIL3 = 40-100 cm depth (°C)
#   TSOIL4 = 100-200 cm depth (°C)
#
# Most useful for planting:
#   TSOIL1 — surface germination conditions
#   TSOIL2 — root zone at seeding depth
#
# Data source: NASA POWER (MERRA2)
# Same source as our climate data — fully consistent.
#
# Run: python src/soil_temperature.py

import os
import sys
import requests
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

LATITUDE      = CONFIG["latitude"]
LONGITUDE     = CONFIG["longitude"]
LOCATION_NAME = CONFIG["location_name"]
NASA_URL      = CONFIG["nasa_power_url"]
OUTPUT_FILE   = CONFIG["soil_temp_clean"]

# Date range — match our existing climate data
START_DATE = "20000101"
END_DATE   = "20251231"

# NASA POWER soil temperature variables
SOIL_VARIABLES = ",".join([
    "TSOIL1",   # 0-10 cm   surface layer
    "TSOIL2",   # 10-40 cm  shallow root zone
    "TSOIL3",   # 40-100 cm deep root zone
])

# NASA missing value
MISSING_VALUE = -999.0

# Month names for display
MONTH_NAMES = {
    1:"January",  2:"February", 3:"March",    4:"April",
    5:"May",      6:"June",     7:"July",     8:"August",
    9:"September",10:"October", 11:"November",12:"December"
}

# -------------------------------------------------------
# STEP 1 - FETCH SOIL TEMPERATURE FROM NASA POWER
# -------------------------------------------------------

print("=" * 60)
print("SOIL TEMPERATURE — NASA POWER")
print(f"Location: {LOCATION_NAME}")
print(f"Coords:   {LATITUDE:.4f}, {LONGITUDE:.4f}")
print(f"Period:   {START_DATE} to {END_DATE}")
print(f"Source:   NASA POWER (MERRA2) — same as climate data")
print("=" * 60)
print()
print("Sending request to NASA POWER...")
print("This may take 30-60 seconds...")

params = {
    "latitude"   : LATITUDE,
    "longitude"  : LONGITUDE,
    "start"      : START_DATE,
    "end"        : END_DATE,
    "community"  : "AG",
    "parameters" : SOIL_VARIABLES,
    "format"     : "JSON",
    "header"     : "true",
}

response = requests.get(NASA_URL, params=params, timeout=120)

if response.status_code != 200:
    print(f"ERROR: Status {response.status_code}")
    print(response.text[:500])
    sys.exit(1)

print(f"SUCCESS (status: {response.status_code})")

# -------------------------------------------------------
# STEP 2 - PARSE RESPONSE
# -------------------------------------------------------

data       = response.json()
parameters = data["properties"]["parameter"]

print(f"Variables received: {list(parameters.keys())}")

# Check which variables we actually got
has_tsoil1 = "TSOIL1" in parameters
has_tsoil2 = "TSOIL2" in parameters
has_tsoil3 = "TSOIL3" in parameters

print(f"  TSOIL1 (0-10cm):   {'yes' if has_tsoil1 else 'NO'}")
print(f"  TSOIL2 (10-40cm):  {'yes' if has_tsoil2 else 'NO'}")
print(f"  TSOIL3 (40-100cm): {'yes' if has_tsoil3 else 'NO'}")

# -------------------------------------------------------
# STEP 3 - BUILD DATAFRAME
# -------------------------------------------------------

dates = list(parameters["TSOIL1"].keys())
print(f"\nTotal days received: {len(dates)}")

df = pd.DataFrame({"date": dates})
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# Add soil temperature columns
if has_tsoil1:
    df["soil_temp_0_10cm"]   = list(parameters["TSOIL1"].values())
if has_tsoil2:
    df["soil_temp_10_40cm"]  = list(parameters["TSOIL2"].values())
if has_tsoil3:
    df["soil_temp_40_100cm"] = list(parameters["TSOIL3"].values())

# Replace NASA missing value with NaN
df = df.replace(MISSING_VALUE, np.nan)

# Add year, month, day
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day

# Primary soil column — use shallowest available layer
# Most useful for seed germination
PRIMARY_SOIL_COL = "soil_temp_0_10cm" \
    if has_tsoil1 else "soil_temp_10_40cm"

print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nMissing values per column:")
print(df.isnull().sum())

# -------------------------------------------------------
# STEP 4 - MONTHLY SOIL TEMPERATURE STATISTICS
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"MONTHLY SOIL TEMPERATURE STATISTICS")
print(f"Based on {START_DATE[:4]}-{END_DATE[:4]} data")
print(f"{'='*60}")
print()

# Complete years only for statistics
days_per_year  = df.groupby("year").size()
complete_years = days_per_year[days_per_year >= 350].index
df_complete    = df[df["year"].isin(complete_years)]

monthly_soil = df_complete.groupby("month").agg(
    soil_surface_avg = (PRIMARY_SOIL_COL, "mean"),
    soil_surface_min = (PRIMARY_SOIL_COL, "min"),
    soil_surface_max = (PRIMARY_SOIL_COL, "max"),
).round(1)

# Add air temperature for comparison
clean_file = CONFIG["clean_data_file"]
has_air    = os.path.exists(clean_file)

if has_air:
    df_air = pd.read_csv(clean_file)
    df_air["date"] = pd.to_datetime(df_air["date"])

    # Filter to same period
    df_air = df_air[df_air["year"].isin(complete_years)]

    air_monthly = df_air.groupby("month")["temp_avg"].mean().round(1)

    print(f"{'Month':<12} {'Air Temp':>10} "
          f"{'Soil 0-10cm':>13} {'Lag':>8}")
    print(f"{'':12} {'(°C avg)':>10} "
          f"{'(°C avg)':>13} {'Air-Soil':>8}")
    print("-" * 48)

    for month_num in range(1, 13):
        air_avg  = air_monthly.get(month_num, np.nan)
        soil_avg = monthly_soil.loc[month_num, "soil_surface_avg"] \
            if month_num in monthly_soil.index else np.nan
        lag      = air_avg - soil_avg \
            if not np.isnan(soil_avg) else np.nan

        lag_str = f"{lag:+.1f}°C" if not np.isnan(lag) else "N/A"

        print(f"{MONTH_NAMES[month_num]:<12} "
              f"{air_avg:>9.1f}°C "
              f"{soil_avg:>11.1f}°C "
              f"{lag_str:>8}")
else:
    print(f"{'Month':<12} {'Soil 0-10cm':>13} "
          f"{'Min':>8} {'Max':>8}")
    print("-" * 44)
    for month_num, row in monthly_soil.iterrows():
        print(f"{MONTH_NAMES[month_num]:<12} "
              f"{row['soil_surface_avg']:>11.1f}°C "
              f"{row['soil_surface_min']:>7.1f}°C "
              f"{row['soil_surface_max']:>7.1f}°C")

# -------------------------------------------------------
# STEP 5 - SPRING SOIL WARMING TREND
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"SPRING SOIL WARMING TREND")
print(f"Are springs arriving earlier? Is soil warming sooner?")
print(f"{'='*60}")
print()

# For each year find the first date soil exceeds 10°C
# and stays there for 5+ days (spring onset proxy)
SPRING_THRESHOLD = 10.0  # °C — general spring warmth indicator

spring_onset = []

for year in sorted(complete_years):
    year_data = df_complete[
        (df_complete["year"] == year) &
        (df_complete["month"].between(1, 6))
    ].reset_index(drop=True)

    if len(year_data) < 30:
        continue

    consec = 0
    found  = None

    for _, row in year_data.iterrows():
        val = row[PRIMARY_SOIL_COL]
        if pd.isna(val):
            consec = 0
            continue
        if val >= SPRING_THRESHOLD:
            consec += 1
            if consec >= 5:
                found = row["date"] - pd.Timedelta(days=4)
                break
        else:
            consec = 0

    if found is not None:
        spring_onset.append({
            "year"       : year,
            "onset_date" : found,
            "day_of_year": found.dayofyear,
        })

if spring_onset:
    onset_df  = pd.DataFrame(spring_onset)
    avg_doy   = onset_df["day_of_year"].mean()
    std_doy   = onset_df["day_of_year"].std()

    def doy_to_str(doy):
        return (pd.Timestamp("2024-01-01") +
                pd.Timedelta(days=int(doy) - 1)).strftime("%B %d")

    print(f"Soil spring onset (>{SPRING_THRESHOLD}°C for 5+ days):")
    print(f"  Average onset date: {doy_to_str(avg_doy)} "
          f"(day {avg_doy:.0f} ± {std_doy:.0f} days)")
    print()

    # Check trend
    from scipy import stats
    x = onset_df["year"].values
    y = onset_df["day_of_year"].values
    slope, _, _, p_value, _ = stats.linregress(x, y)

    print(f"  Trend: {slope:.2f} days per year "
          f"({slope*10:.1f} days per decade)")
    if p_value < 0.05:
        direction = "earlier" if slope < 0 else "later"
        print(f"  Soil spring is arriving "
              f"{direction} over time (p={p_value:.3f})")
    else:
        print(f"  No statistically significant trend "
              f"(p={p_value:.3f})")

    # Early vs recent comparison
    early_onset  = onset_df[
        onset_df["year"] <= 2012
    ]["day_of_year"].mean()
    recent_onset = onset_df[
        onset_df["year"] > 2012
    ]["day_of_year"].mean()

    print()
    print(f"  Early period  (2000-2012): "
          f"avg onset {doy_to_str(early_onset)}")
    print(f"  Recent period (2013-2025): "
          f"avg onset {doy_to_str(recent_onset)}")
    print(f"  Shift: {early_onset - recent_onset:.1f} days earlier")

# -------------------------------------------------------
# STEP 6 - CROP PLANTING READINESS
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"CROP PLANTING READINESS FROM SOIL TEMPERATURE")
print(f"When does soil reach germination temperature")
print(f"for 5+ consecutive days?")
print(f"{'='*60}")
print()

from crop_model import CROPS

# Get germination thresholds
thresholds = {}
for crop_key, crop in CROPS.items():
    if "germination" in crop:
        thresholds[crop_key] = {
            "name"  : crop["name"],
            "min_c" : crop["germination"]["soil_temp_min_c"],
        }

all_readiness = []

print(f"{'Crop':<14} {'Min Soil':>9} {'Avg Ready':>11} "
      f"{'Early':>9} {'Median':>9} {'Safe':>9}")
print(f"{'':14} {'Temp':>9} {'Date':>11} "
      f"{'(10th%)':>9} {'(50th%)':>9} {'(90th%)':>9}")
print("-" * 64)

for crop_key, thresh in thresholds.items():
    crop_name = thresh["name"]
    min_temp  = thresh["min_c"]
    ready_doys = []

    for year in sorted(complete_years):
        spring = df_complete[
            (df_complete["year"] == year) &
            (df_complete["month"].between(1, 7))
        ].reset_index(drop=True)

        if len(spring) < 30:
            continue

        consec = 0
        found  = None

        for _, row in spring.iterrows():
            val = row[PRIMARY_SOIL_COL]
            if pd.isna(val):
                consec = 0
                continue
            if val >= min_temp:
                consec += 1
                if consec >= 5:
                    found = row["date"] - pd.Timedelta(days=4)
                    break
            else:
                consec = 0

        if found is not None:
            ready_doys.append(found.dayofyear)
            all_readiness.append({
                "crop_key"   : crop_key,
                "crop_name"  : crop_name,
                "min_soil_c" : min_temp,
                "year"       : year,
                "day_of_year": found.dayofyear,
                "ready_date" : str(found.date()),
            })

    if not ready_doys:
        print(f"{crop_name:<14} {min_temp:>8.0f}°C  "
              f"{'never reached':>11}")
        continue

    doys    = pd.Series(ready_doys)
    avg_doy = doys.mean()
    p10_doy = doys.quantile(0.10)
    p50_doy = doys.quantile(0.50)
    p90_doy = doys.quantile(0.90)

    print(f"{crop_name:<14} "
          f"{min_temp:>8.0f}°C  "
          f"{doy_to_str(avg_doy):>11} "
          f"{doy_to_str(p10_doy):>9} "
          f"{doy_to_str(p50_doy):>9} "
          f"{doy_to_str(p90_doy):>9}")

# -------------------------------------------------------
# STEP 7 - COMPARE WITH AIR-BASED RECOMMENDATIONS
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"HOW SOIL TEMPERATURE REFINES OUR RECOMMENDATIONS")
print(f"{'='*60}")
print()
print(f"Our risk model recommended dates based on AIR temperature.")
print(f"Soil temperature shows when germination is actually possible.")
print()
print(f"{'Crop':<14} {'Air-Based':>12} {'Soil-Based':>12} "
      f"{'Adjustment':>12}")
print("-" * 54)

# Air-based best dates from our risk model
air_recommendations = {
    "potato"    : ("Mar 01", 61),
    "tomato"    : ("Apr 10", 99),
    "onion"     : ("Mar 01", 60),
    "cucumber"  : ("May 01", 91),
    "pepper"    : ("Apr 20", 110),
    "watermelon": ("May 01", 121),
    "sunflower" : ("Apr 10", 101),
    "corn"      : ("Apr 20", 111),
}

# Get soil-based median dates
soil_medians = {}
if all_readiness:
    rdf = pd.DataFrame(all_readiness)
    for crop_key in CROPS.keys():
        crop_data = rdf[rdf["crop_key"] == crop_key]
        if not crop_data.empty:
            median_doy = crop_data["day_of_year"].median()
            soil_medians[crop_key] = median_doy

for crop_key in CROPS.keys():
    crop_name = CROPS[crop_key]["name"]
    air_rec   = air_recommendations.get(crop_key)
    soil_doy  = soil_medians.get(crop_key)

    if air_rec is None:
        continue

    air_str  = air_rec[0]
    air_doy  = air_rec[1]

    if soil_doy is not None:
        soil_str = doy_to_str(soil_doy)
        diff     = soil_doy - air_doy
        if abs(diff) <= 5:
            adj = "No change"
        elif diff > 0:
            adj = f"Plant {diff:.0f} days later"
        else:
            adj = f"Plant {abs(diff):.0f} days earlier"
    else:
        soil_str = "insufficient data"
        adj      = "—"

    print(f"{crop_name:<14} "
          f"{air_str:>12} "
          f"{soil_str:>12} "
          f"{adj:>12}")

# -------------------------------------------------------
# STEP 8 - SAVE
# -------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSoil temperature data saved to: {OUTPUT_FILE}")
print(f"Total records: {len(df):,}")

if all_readiness:
    rdf   = pd.DataFrame(all_readiness)
    rfile = "data/processed/soil_readiness_dates.csv"
    rdf.to_csv(rfile, index=False)
    print(f"Soil readiness dates saved to:  {rfile}")

print("\n=== SOIL TEMPERATURE ANALYSIS COMPLETE ===")
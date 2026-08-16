# src/data_cleaning.py
#
# Investigate and clean our NASA POWER dataset.
#
# We are NOT immediately deleting anything suspicious.
# First we LOOK. Then we decide.
#
# What this script does:
# 1. Loads the CSV we downloaded
# 2. Inspects the data carefully
# 3. Checks for missing values
# 4. Checks for suspicious values
# 5. Checks for date gaps
# 6. Produces a clean version we can trust

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE    = CONFIG["raw_data_file"]
OUTPUT_FILE   = CONFIG["clean_data_file"]
LOCATION_NAME = CONFIG["location_name"]

TEMP_MIN_PLAUSIBLE = -40.0
TEMP_MAX_PLAUSIBLE =  50.0

# -------------------------------------------------------
# STEP 1 - LOAD THE DATA
# -------------------------------------------------------

print("Loading data...")
print("-" * 60)

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

print(f"Location:    {LOCATION_NAME}")
print(f"Rows loaded: {len(df):,}")
print(f"Columns:     {list(df.columns)}")
print(f"Date range:  {df['date'].min()} to {df['date'].max()}")

# -------------------------------------------------------
# STEP 2 - CHECK FOR DATE GAPS
# -------------------------------------------------------

print("\n--- DATE GAP CHECK ---")

full_date_range = pd.date_range(
    start = df["date"].min(),
    end   = df["date"].max(),
    freq  = "D"
)

missing_dates = full_date_range.difference(df["date"])

if len(missing_dates) == 0:
    print("No date gaps found — every day is present ✓")
else:
    print(f"WARNING: {len(missing_dates)} missing dates found!")
    print(missing_dates)

# -------------------------------------------------------
# STEP 3 - CHECK FOR MISSING VALUES
# -------------------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count"   : missing,
    "missing_percent" : missing_pct
})

# Only show columns that have missing values
has_missing = missing_report[missing_report["missing_count"] > 0]
if has_missing.empty:
    print("No missing values found ✓")
else:
    print(has_missing)

# -------------------------------------------------------
# STEP 4 - SOLAR RADIATION MISSING BY YEAR
# -------------------------------------------------------

if "solar_rad" in df.columns:
    print("\n--- SOLAR RADIATION MISSING DATA BY YEAR ---")
    solar_missing    = df[df["solar_rad"].isnull()]
    missing_by_year  = solar_missing.groupby("year").size()
    if missing_by_year.empty:
        print("No missing solar radiation data ✓")
    else:
        print(missing_by_year)

# -------------------------------------------------------
# STEP 5 - TEMPERATURE RANGE CHECK
# -------------------------------------------------------

print("\n--- TEMPERATURE RANGE CHECK ---")

print(f"temp_avg: min={df['temp_avg'].min():.2f}°C  "
      f"max={df['temp_avg'].max():.2f}°C")
print(f"temp_max: min={df['temp_max'].min():.2f}°C  "
      f"max={df['temp_max'].max():.2f}°C")
print(f"temp_min: min={df['temp_min'].min():.2f}°C  "
      f"max={df['temp_min'].max():.2f}°C")

suspicious_high = df[df["temp_max"] > TEMP_MAX_PLAUSIBLE]
suspicious_low  = df[df["temp_min"] < TEMP_MIN_PLAUSIBLE]

print(f"\nDays above {TEMP_MAX_PLAUSIBLE}°C: {len(suspicious_high)}")
print(f"Days below {TEMP_MIN_PLAUSIBLE}°C: {len(suspicious_low)}")

if len(suspicious_high) > 0:
    print("\nSuspiciously hot days:")
    print(suspicious_high[["date","temp_avg","temp_max","temp_min"]])

if len(suspicious_low) > 0:
    print("\nSuspiciously cold days:")
    print(suspicious_low[["date","temp_avg","temp_max","temp_min"]])

# -------------------------------------------------------
# STEP 6 - LOGICAL CONSISTENCY CHECK
# -------------------------------------------------------

print("\n--- LOGICAL CONSISTENCY CHECK ---")

impossible = df[df["temp_min"] > df["temp_max"]]
print(f"Rows where temp_min > temp_max: {len(impossible)}")

if len(impossible) > 0:
    print(impossible[["date","temp_avg","temp_max","temp_min"]])

# -------------------------------------------------------
# STEP 7 - PRECIPITATION CHECK
# -------------------------------------------------------

print("\n--- PRECIPITATION CHECK ---")

print(f"Max single day rainfall: {df['precipitation'].max():.2f} mm")
print(f"Days with zero rain:     {(df['precipitation'] == 0).sum():,}")
print(f"Days with rain > 50mm:   {(df['precipitation'] > 50).sum()}")
print(f"Days with rain > 100mm:  {(df['precipitation'] > 100).sum()}")

print("\nTop 5 wettest days:")
print(df.nlargest(5, "precipitation")[
    ["date", "precipitation", "temp_avg"]
])

# -------------------------------------------------------
# STEP 8 - SOIL MOISTURE CHECK (NEW VARIABLES)
# -------------------------------------------------------

if "soil_wet_root" in df.columns:
    print("\n--- SOIL MOISTURE CHECK ---")
    print(f"Root zone wetness range: "
          f"{df['soil_wet_root'].min():.3f} to "
          f"{df['soil_wet_root'].max():.3f}")
    print(f"(0 = completely dry, 1 = fully saturated)")

    # Count drought periods
    drought_days = (df["soil_wet_root"] < 0.2).sum()
    print(f"Days with low root moisture (<0.2): {drought_days:,}")

if "evapotranspiration" in df.columns:
    print(f"\nEvapotranspiration range: "
          f"{df['evapotranspiration'].min():.2f} to "
          f"{df['evapotranspiration'].max():.2f} mm/day")

# -------------------------------------------------------
# STEP 9 - GDD CHECK
# -------------------------------------------------------

if "gdd_base_10" in df.columns:
    print("\n--- GROWING DEGREE DAYS CHECK ---")
    annual_gdd = df.groupby("year")["gdd_base_10"].sum()
    print(f"Annual GDD (base 10°C):")
    print(f"  Average: {annual_gdd.mean():.0f}")
    print(f"  Min:     {annual_gdd.min():.0f} ({annual_gdd.idxmin()})")
    print(f"  Max:     {annual_gdd.max():.0f} ({annual_gdd.idxmax()})")

# -------------------------------------------------------
# STEP 10 - BASIC STATISTICS
# -------------------------------------------------------

print("\n--- BASIC STATISTICS ---")
core_cols = [
    "temp_avg", "temp_max", "temp_min",
    "precipitation", "humidity"
]
available_core = [c for c in core_cols if c in df.columns]
print(df[available_core].describe().round(2))

# -------------------------------------------------------
# STEP 11 - IDENTIFY COMPLETE VS PARTIAL YEARS
# -------------------------------------------------------

print("\n--- YEAR COMPLETENESS ---")

days_per_year   = df.groupby("year").size()
complete_years  = days_per_year[days_per_year >= 350]
partial_years   = days_per_year[days_per_year < 350]

print(f"Complete years (>=350 days): "
      f"{len(complete_years)} "
      f"({complete_years.index.min()}-{complete_years.index.max()})")

if not partial_years.empty:
    print(f"Partial years excluded from annual stats:")
    for yr, days in partial_years.items():
        print(f"  {yr}: {days} days")

# -------------------------------------------------------
# STEP 12 - CLEAN AND SAVE
# -------------------------------------------------------

print("\n--- CLEANING DATA ---")

df_clean = df.copy()

# Add solar radiation availability flag if column exists
if "solar_rad" in df_clean.columns:
    df_clean["solar_rad_available"] = df_clean["solar_rad"].notna()
    print(f"Rows with solar data:    "
          f"{df_clean['solar_rad_available'].sum():,}")
    print(f"Rows without solar data: "
          f"{(~df_clean['solar_rad_available']).sum():,}")

# Report overall data quality
total_cells    = len(df_clean) * len(df_clean.columns)
missing_cells  = df_clean.isnull().sum().sum()
complete_pct   = (1 - missing_cells / total_cells) * 100
print(f"\nOverall data completeness: {complete_pct:.1f}%")

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nClean data saved to: {OUTPUT_FILE}")
print(f"Total records: {len(df_clean):,}")
print(f"Total columns: {len(df_clean.columns)}")
print("\nData cleaning complete!")
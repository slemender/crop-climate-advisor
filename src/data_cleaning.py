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

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE  = "data/raw/vojvodina_nasa_power_daily.csv"
OUTPUT_FILE = "data/processed/vojvodina_clean.csv"

# -------------------------------------------------------
# STEP 1 - LOAD THE DATA
# -------------------------------------------------------

print("Loading data...")
print("-" * 60)

df = pd.read_csv(INPUT_FILE)

# Tell pandas the date column is a real date
df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded:    {len(df)}")
print(f"Columns:        {list(df.columns)}")
print(f"Date range:     {df['date'].min()} to {df['date'].max()}")

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
    print("No date gaps found - every day is present ✓")
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

print(missing_report)

# -------------------------------------------------------
# STEP 4 - INVESTIGATE THE SOLAR RADIATION GAP
# -------------------------------------------------------

print("\n--- SOLAR RADIATION MISSING DATA BY YEAR ---")

solar_missing    = df[df["solar_rad"].isnull()]
missing_by_year  = solar_missing.groupby("year").size()
print(missing_by_year)

# -------------------------------------------------------
# STEP 5 - CHECK FOR SUSPICIOUS TEMPERATURE VALUES
# -------------------------------------------------------

print("\n--- TEMPERATURE RANGE CHECK ---")

print(f"temp_avg:  min={df['temp_avg'].min():.2f}°C   "
      f"max={df['temp_avg'].max():.2f}°C")
print(f"temp_max:  min={df['temp_max'].min():.2f}°C   "
      f"max={df['temp_max'].max():.2f}°C")
print(f"temp_min:  min={df['temp_min'].min():.2f}°C   "
      f"max={df['temp_min'].max():.2f}°C")

TEMP_MIN_PLAUSIBLE = -40.0
TEMP_MAX_PLAUSIBLE =  50.0

suspicious_high = df[df["temp_max"] > TEMP_MAX_PLAUSIBLE]
suspicious_low  = df[df["temp_min"] < TEMP_MIN_PLAUSIBLE]

print(f"\nDays with temp_max above {TEMP_MAX_PLAUSIBLE}°C: "
      f"{len(suspicious_high)}")
print(f"Days with temp_min below {TEMP_MIN_PLAUSIBLE}°C: "
      f"{len(suspicious_low)}")

if len(suspicious_high) > 0:
    print("\nSuspiciously hot days:")
    print(suspicious_high[["date","temp_avg","temp_max","temp_min"]])

if len(suspicious_low) > 0:
    print("\nSuspiciously cold days:")
    print(suspicious_low[["date","temp_avg","temp_max","temp_min"]])

# -------------------------------------------------------
# STEP 6 - CHECK FOR IMPOSSIBLE COMBINATIONS
# -------------------------------------------------------

print("\n--- LOGICAL CONSISTENCY CHECK ---")

impossible = df[df["temp_min"] > df["temp_max"]]
print(f"Rows where temp_min > temp_max: {len(impossible)}")

if len(impossible) > 0:
    print(impossible[["date","temp_avg","temp_max","temp_min"]])

# -------------------------------------------------------
# STEP 7 - CHECK PRECIPITATION
# -------------------------------------------------------

print("\n--- PRECIPITATION CHECK ---")

print(f"Max single day rainfall: {df['precipitation'].max():.2f} mm")
print(f"Days with zero rain:     {(df['precipitation'] == 0).sum()}")
print(f"Days with rain > 50mm:   {(df['precipitation'] > 50).sum()}")
print(f"Days with rain > 100mm:  {(df['precipitation'] > 100).sum()}")

print("\nTop 5 wettest days:")
print(df.nlargest(5, "precipitation")[
    ["date","precipitation","temp_avg"]
])

# -------------------------------------------------------
# STEP 8 - BASIC STATISTICS SUMMARY
# -------------------------------------------------------

print("\n--- BASIC STATISTICS ---")
print(df[["temp_avg","temp_max","temp_min",
          "precipitation","humidity"]].describe().round(2))

# -------------------------------------------------------
# STEP 9 - CLEAN THE DATA
# -------------------------------------------------------

print("\n--- CLEANING DATA ---")

df_clean = df.copy()

# Add a flag so we always know which rows have solar data
df_clean["solar_rad_available"] = df_clean["solar_rad"].notna()

print(f"Rows with solar radiation data:    "
      f"{df_clean['solar_rad_available'].sum()}")
print(f"Rows without solar radiation data: "
      f"{(~df_clean['solar_rad_available']).sum()}")

# -------------------------------------------------------
# STEP 10 - SAVE CLEAN DATA
# -------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nClean data saved to: {OUTPUT_FILE}")
print(f"Total records: {len(df_clean)}")
print("\nData cleaning complete!")
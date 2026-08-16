# src/data_cleaning.py
#
# Clean and validate NASA POWER climate data.
# File paths read from config.py

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
# LOAD
# -------------------------------------------------------

print("Loading data...")
print("-" * 60)

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

print(f"Location:    {LOCATION_NAME}")
print(f"Rows loaded: {len(df)}")
print(f"Columns:     {list(df.columns)}")
print(f"Date range:  {df['date'].min()} to {df['date'].max()}")

# -------------------------------------------------------
# DATE GAP CHECK
# -------------------------------------------------------

print("\n--- DATE GAP CHECK ---")

full_range    = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="D"
)
missing_dates = full_range.difference(df["date"])

if len(missing_dates) == 0:
    print("No date gaps found — every day is present ✓")
else:
    print(f"WARNING: {len(missing_dates)} missing dates!")
    print(missing_dates)

# -------------------------------------------------------
# MISSING VALUE CHECK
# -------------------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

print(pd.DataFrame({
    "missing_count"   : missing,
    "missing_percent" : missing_pct
}))

# -------------------------------------------------------
# SOLAR RADIATION MISSING BY YEAR
# -------------------------------------------------------

print("\n--- SOLAR RADIATION MISSING DATA BY YEAR ---")
solar_missing  = df[df["solar_rad"].isnull()]
missing_by_year = solar_missing.groupby("year").size()
print(missing_by_year)

# -------------------------------------------------------
# TEMPERATURE RANGE CHECK
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
    print(suspicious_high[["date","temp_avg","temp_max","temp_min"]])
if len(suspicious_low) > 0:
    print(suspicious_low[["date","temp_avg","temp_max","temp_min"]])

# -------------------------------------------------------
# LOGICAL CONSISTENCY CHECK
# -------------------------------------------------------

print("\n--- LOGICAL CONSISTENCY CHECK ---")
impossible = df[df["temp_min"] > df["temp_max"]]
print(f"Rows where temp_min > temp_max: {len(impossible)}")

# -------------------------------------------------------
# PRECIPITATION CHECK
# -------------------------------------------------------

print("\n--- PRECIPITATION CHECK ---")
print(f"Max single day rainfall: {df['precipitation'].max():.2f} mm")
print(f"Days with zero rain:     {(df['precipitation'] == 0).sum()}")
print(f"Days with rain > 50mm:   {(df['precipitation'] > 50).sum()}")

print("\nTop 5 wettest days:")
print(df.nlargest(5, "precipitation")[
    ["date","precipitation","temp_avg"]
])

# -------------------------------------------------------
# BASIC STATISTICS
# -------------------------------------------------------

print("\n--- BASIC STATISTICS ---")
print(df[["temp_avg","temp_max","temp_min",
          "precipitation","humidity"]].describe().round(2))

# -------------------------------------------------------
# CLEAN AND SAVE
# -------------------------------------------------------

print("\n--- CLEANING DATA ---")

df_clean = df.copy()
df_clean["solar_rad_available"] = df_clean["solar_rad"].notna()

print(f"Rows with solar data:    {df_clean['solar_rad_available'].sum()}")
print(f"Rows without solar data: {(~df_clean['solar_rad_available']).sum()}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nClean data saved to: {OUTPUT_FILE}")
print(f"Total records: {len(df_clean)}")
print("\nData cleaning complete!")
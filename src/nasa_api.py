# src/nasa_api.py
#
# Fetch daily climate data from NASA POWER for Vojvodina.
# NASA POWER limits requests to 20 variables maximum.
# We make two separate requests and merge the results.
#
# Run: python src/nasa_api.py

import requests
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

LATITUDE      = CONFIG["latitude"]
LONGITUDE     = CONFIG["longitude"]
LOCATION_NAME = CONFIG["location_name"]
START_DATE    = CONFIG["nasa_start_date"]
END_DATE      = CONFIG["nasa_end_date"]
OUTPUT_FILE   = CONFIG["raw_data_file"]
OUTPUT_FOLDER = os.path.dirname(OUTPUT_FILE)
NASA_URL      = CONFIG["nasa_power_url"]

MISSING_VALUE = -999.0

# -------------------------------------------------------
# VARIABLE GROUPS
# NASA POWER allows maximum 20 variables per request.
# We split into two batches and merge afterwards.
# -------------------------------------------------------

# Batch 1 — Core climate variables (18 variables)
BATCH_1 = ",".join([
    "T2M",               # Average air temp at 2m (°C)
    "T2M_MAX",           # Daily maximum temp (°C)
    "T2M_MIN",           # Daily minimum temp (°C)
    "T2MDEW",            # Dew point temp at 2m (°C)
    "PRECTOTCORR",       # Precipitation corrected (mm/day)
    "RH2M",              # Relative humidity (%)
    "ALLSKY_SFC_SW_DWN", # Solar radiation (MJ/m²/day)
    "WS2M",              # Wind speed at 2m (m/s)
    "TSOIL1",            # Soil temp 0-10cm (°C)
    "TSOIL2",            # Soil temp 10-40cm (°C)
    "TSOIL3",            # Soil temp 40-100cm (°C)
    "CLOUD_AMT",         # Cloud cover (%)
    "FROST_DAYS",        # Frost days flag
    "GDD10",             # GDD base 10°C
    "GDD7_2",            # GDD base 7.2°C
])

# Batch 2 — Soil moisture and additional variables (10 variables)
BATCH_2 = ",".join([
    "GWETTOP",           # Surface soil wetness 0-1 (fraction)
    "GWETROOT",          # Root zone soil wetness 0-1 (fraction)
    "GWETPROF",          # Profile soil wetness 0-1 (fraction)
    "EVLAND",            # Evapotranspiration over land (mm/day)
    "GDD4_4",            # GDD base 4.4°C (onion, cool crops)
    "GDD13_3",           # GDD base 13.3°C (pepper, warm crops)
    "T10M",              # Temperature at 10m height (°C)
    "PRECSNO",           # Snowfall (mm/day water equivalent)
    "QV2M",              # Specific humidity at 2m (g/kg)
    "ALLSKY_SFC_PAR_TOT",# Photosynthetically active radiation
])
# -------------------------------------------------------
# FETCH FUNCTION
# -------------------------------------------------------

def fetch_nasa_batch(variables, batch_name):
    """
    Fetch one batch of variables from NASA POWER.
    Returns a DataFrame or None if failed.
    """
    print(f"\n{batch_name}: {len(variables.split(','))} variables")
    print(f"  Variables: {variables[:80]}...")

    params = {
        "latitude"   : LATITUDE,
        "longitude"  : LONGITUDE,
        "start"      : START_DATE,
        "end"        : END_DATE,
        "community"  : CONFIG["nasa_community"],
        "parameters" : variables,
        "format"     : "JSON",
        "header"     : "true",
    }

    response = requests.get(
        NASA_URL, params=params, timeout=180
    )

    if response.status_code != 200:
        print(f"  ERROR: Status {response.status_code}")
        print(f"  {response.text[:300]}")
        return None

    data       = response.json()
    parameters = data["properties"]["parameter"]
    dates      = list(list(parameters.values())[0].keys())

    print(f"  SUCCESS: {len(parameters)} variables, "
          f"{len(dates)} days")

    # Build DataFrame from this batch
    rows = {"date": dates}
    for var_name, values in parameters.items():
        rows[var_name] = list(values.values())

    df = pd.DataFrame(rows)
    df = df.replace(MISSING_VALUE, np.nan)

    return df, dates


# -------------------------------------------------------
# MAIN FETCH
# -------------------------------------------------------

print(f"Requesting NASA POWER data for {LOCATION_NAME}")
print(f"Coordinates: {LATITUDE:.4f}, {LONGITUDE:.4f}")
print(f"Date range:  {START_DATE} to {END_DATE}")
print(f"Strategy:    Two batches (max 20 variables each)")
print("-" * 60)

# Fetch both batches
result1 = fetch_nasa_batch(BATCH_1, "Batch 1 (core climate)")
result2 = fetch_nasa_batch(BATCH_2, "Batch 2 (soil moisture)")

if result1 is None:
    print("\nFATAL: Batch 1 failed. Cannot continue.")
    sys.exit(1)

df1, dates = result1

# Merge batch 2 if available
if result2 is not None:
    df2, _ = result2

    # Merge on date
    df = pd.merge(df1, df2, on="date", how="left")
    print(f"\nMerged both batches successfully.")
else:
    print("\nWARNING: Batch 2 failed. Continuing with Batch 1 only.")
    df = df1

# -------------------------------------------------------
# RENAME COLUMNS TO FRIENDLY NAMES
# -------------------------------------------------------

rename_map = {
    # Temperature
    "T2M"               : "temp_avg",
    "T2M_MAX"           : "temp_max",
    "T2M_MIN"           : "temp_min",
    "T2MDEW"            : "dew_point",

    # Precipitation
    "PRECTOTCORR"       : "precipitation",
    "PRECSNO"           : "snowfall",

    # Humidity
    "RH2M"              : "humidity",
    "QV2M"              : "specific_humidity",

    # Solar
    "ALLSKY_SFC_SW_DWN" : "solar_rad",
    "ALLSKY_SFC_PAR_TOT": "par",

    # Wind
    "WS2M"              : "wind_speed",

    # Soil temperature
    "TSOIL1"            : "soil_temp_0_10cm",
    "TSOIL2"            : "soil_temp_10_40cm",
    "TSOIL3"            : "soil_temp_40_100cm",

    # Cloud and frost
    "CLOUD_AMT"         : "cloud_cover",
    "FROST_DAYS"        : "nasa_frost_flag",

    # GDD
    "GDD10"             : "gdd_base_10",
    "GDD7_2"            : "gdd_base_7",
    "GDD4_4"            : "gdd_base_4",
    "GDD13_3"           : "gdd_base_13",

    # Soil wetness (GWET variables — these work)
    "GWETTOP"           : "soil_wet_surface",
    "GWETROOT"          : "soil_wet_root",
    "GWETPROF"          : "soil_wet_profile",

    # Evapotranspiration
    "EVLAND"            : "evapotranspiration",

    # Additional
    "T10M"              : "temp_10m",
}

# Only rename columns that exist
rename_existing = {
    k: v for k, v in rename_map.items()
    if k in df.columns
}
df = df.rename(columns=rename_existing)

# -------------------------------------------------------
# PARSE DATES AND ADD CALENDAR COLUMNS
# -------------------------------------------------------

df["date"]        = pd.to_datetime(df["date"], format="%Y%m%d")
df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["day"]         = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

# -------------------------------------------------------
# REPORT
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"DATASET SUMMARY")
print(f"{'='*60}")
print(f"Shape:      {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['date'].min().date()} "
      f"to {df['date'].max().date()}")

print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    missing = df[col].isna().sum()
    pct     = missing / len(df) * 100
    status  = f"{missing:,} missing ({pct:.1f}%)" \
        if missing > 0 else "complete"
    print(f"  {col:<30} {status}")

# Key statistics
print(f"\nKey statistics:")
print(f"  Temperature range: "
      f"{df['temp_min'].min():.1f}°C to "
      f"{df['temp_max'].max():.1f}°C")
print(f"  Max daily rainfall: "
      f"{df['precipitation'].max():.1f} mm")

if "soil_moisture_pctl" in df.columns:
    print(f"  Soil moisture percentile avg: "
          f"{df['soil_moisture_pctl'].mean():.1f}%")
    drought_days = (df["soil_moisture_pctl"] < 10).sum()
    print(f"  Drought days (below 10th pct): "
          f"{drought_days:,}")

if "gdd_base_10" in df.columns:
    annual_gdd = df.groupby("year")["gdd_base_10"].sum().mean()
    print(f"  Avg annual GDD (base 10°C): {annual_gdd:.0f}")

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")
print(f"Records:  {len(df):,}")
print(f"Columns:  {len(df.columns)}")
print("\nDone!")
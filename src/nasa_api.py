# src/nasa_api.py
#
# Fetch daily climate data from NASA POWER for Vojvodina.
# Location and date range read from .env via config.py
#
# Variables fetched:
#   Temperature:       T2M, T2M_MAX, T2M_MIN, T2MDEW
#   Precipitation:     PRECTOTCORR, PRECSNO
#   Humidity:          RH2M, QV2M
#   Solar/Radiation:   ALLSKY_SFC_SW_DWN, ALLSKY_SFC_PAR_TOT
#   Wind:              WS2M
#   Soil Temperature:  TSOIL1, TSOIL2, TSOIL3
#   Soil Moisture:     GWETTOP, GWETROOT, GWETPROF
#                      SFMC, RZMC, RZMC_PRCNTL
#   Evapotranspiration:EVLAND
#   GDD (pre-calc):    GDD4_4, GDD7_2, GDD10, GDD13_3
#   Frost:             FROST_DAYS
#   Cloud:             CLOUD_AMT
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

# -------------------------------------------------------
# VARIABLES TO FETCH
# -------------------------------------------------------
# Organised by category for readability.
# Each variable is explained with its unit.

VARIABLES = ",".join([

    # --- TEMPERATURE ---
    "T2M",          # Average air temp at 2m (°C)
    "T2M_MAX",      # Daily maximum temp (°C)
    "T2M_MIN",      # Daily minimum temp (°C)
    "T2MDEW",       # Dew point temperature at 2m (°C)
                    # When air cools to dew point, frost risk rises
                    # Also indicates humidity stress on crops

    # --- PRECIPITATION ---
    "PRECTOTCORR",  # Total precipitation corrected (mm/day)
    "PRECSNO",      # Snowfall (mm/day water equivalent)
                    # Snowpack is a reservoir for spring soil moisture

    # --- HUMIDITY ---
    "RH2M",         # Relative humidity at 2m (%)
    "QV2M",         # Specific humidity at 2m (g/kg)
                    # Specific humidity unlike RH does not depend on temp

    # --- SOLAR RADIATION ---
    "ALLSKY_SFC_SW_DWN",  # Total shortwave radiation (MJ/m²/day)
    "ALLSKY_SFC_PAR_TOT", # Photosynthetically Active Radiation (MJ/m²/day)
                          # PAR is the specific light wavelength plants use
                          # for photosynthesis — more direct crop relevance
                          # than total solar radiation

    # --- WIND ---
    "WS2M",         # Wind speed at 2m (m/s)

    # --- SOIL TEMPERATURE ---
    "TSOIL1",       # Soil temp 0-10cm (°C)
    "TSOIL2",       # Soil temp 10-40cm (°C)
    "TSOIL3",       # Soil temp 40-100cm (°C)

    # --- SOIL MOISTURE ---
    "GWETTOP",      # Surface soil wetness 0=dry 1=saturated (fraction)
    "GWETROOT",     # Root zone soil wetness (fraction)
                    # More relevant than surface for established crops
    "GWETPROF",     # Profile soil wetness full column (fraction)
    "SFMC",         # Surface soil moisture content (m³/m³)
    "RZMC",         # Root zone soil moisture content (m³/m³)
    "RZMC_PRCNTL",  # Root zone moisture percentile (%)
                    # 10% = drier than 90% of historical records
                    # This is a direct drought index — very useful

    # --- EVAPOTRANSPIRATION ---
    "EVLAND",       # Actual evapotranspiration over land (mm/day)
                    # Combined with rain gives true water balance

    # --- PRE-CALCULATED GROWING DEGREE DAYS ---
    # NASA calculates these directly — saves us computing them
    "GDD4_4",       # GDD base 4.4°C (onion, cool crops)
    "GDD7_2",       # GDD base 7.2°C (potato)
    "GDD10",        # GDD base 10°C  (tomato, corn, cucumber)
    "GDD13_3",      # GDD base 13.3°C (pepper, warm crops)

    # --- FROST ---
    "FROST_DAYS",   # Days with min temp at or below 0°C
                    # NASA pre-calculated — cross-checks our analysis

    # --- CLOUD COVER ---
    "CLOUD_AMT",    # Total cloud amount (%)
                    # Cloudy nights are warmer — affects frost risk
                    # Cloudy days reduce PAR and crop growth rate
])

MISSING_VALUE = -999.0

# -------------------------------------------------------
# SEND REQUEST TO NASA POWER
# -------------------------------------------------------

params = {
    "latitude"   : LATITUDE,
    "longitude"  : LONGITUDE,
    "start"      : START_DATE,
    "end"        : END_DATE,
    "community"  : CONFIG["nasa_community"],
    "parameters" : VARIABLES,
    "format"     : "JSON",
    "header"     : "true",
}

print(f"Requesting NASA POWER data for {LOCATION_NAME}")
print(f"Coordinates: {LATITUDE:.4f}, {LONGITUDE:.4f}")
print(f"Date range:  {START_DATE} to {END_DATE}")
print(f"Variables:   {len(VARIABLES.split(','))}")
print("This may take 60-90 seconds for extended variable set...")
print("-" * 60)

response = requests.get(NASA_URL, params=params, timeout=180)

# -------------------------------------------------------
# CHECK RESPONSE
# -------------------------------------------------------

if response.status_code != 200:
    print(f"ERROR — NASA returned status: {response.status_code}")
    print(response.text[:500])
    sys.exit(1)

print(f"SUCCESS (status: {response.status_code})")

data       = response.json()
parameters = data["properties"]["parameter"]

print(f"Variables received: {len(parameters)}")

# Check which variables came back
expected = VARIABLES.split(",")
missing  = [v for v in expected if v not in parameters]
if missing:
    print(f"WARNING: These variables were not returned: {missing}")

# -------------------------------------------------------
# BUILD DATAFRAME
# -------------------------------------------------------

dates = list(parameters["T2M"].keys())
print(f"Total days: {len(dates)}")

def get_values(var_name):
    """
    Safely extract values for a variable.
    Returns list of NaN if variable not available.
    """
    if var_name in parameters:
        return list(parameters[var_name].values())
    else:
        print(f"  Note: {var_name} not available — filling with NaN")
        return [np.nan] * len(dates)


rows = {
    # Dates
    "date"               : dates,

    # Temperature
    "temp_avg"           : get_values("T2M"),
    "temp_max"           : get_values("T2M_MAX"),
    "temp_min"           : get_values("T2M_MIN"),
    "dew_point"          : get_values("T2MDEW"),

    # Precipitation
    "precipitation"      : get_values("PRECTOTCORR"),
    "snowfall"           : get_values("PRECSNO"),

    # Humidity
    "humidity"           : get_values("RH2M"),
    "specific_humidity"  : get_values("QV2M"),

    # Solar
    "solar_rad"          : get_values("ALLSKY_SFC_SW_DWN"),
    "par"                : get_values("ALLSKY_SFC_PAR_TOT"),

    # Wind
    "wind_speed"         : get_values("WS2M"),

    # Soil temperature
    "soil_temp_0_10cm"   : get_values("TSOIL1"),
    "soil_temp_10_40cm"  : get_values("TSOIL2"),
    "soil_temp_40_100cm" : get_values("TSOIL3"),

    # Soil moisture
    "soil_wet_surface"   : get_values("GWETTOP"),
    "soil_wet_root"      : get_values("GWETROOT"),
    "soil_wet_profile"   : get_values("GWETPROF"),
    "soil_moisture_surf" : get_values("SFMC"),
    "soil_moisture_root" : get_values("RZMC"),
    "soil_moisture_pctl" : get_values("RZMC_PRCNTL"),

    # Evapotranspiration
    "evapotranspiration" : get_values("EVLAND"),

    # Pre-calculated GDD
    "gdd_base_4"         : get_values("GDD4_4"),
    "gdd_base_7"         : get_values("GDD7_2"),
    "gdd_base_10"        : get_values("GDD10"),
    "gdd_base_13"        : get_values("GDD13_3"),

    # Frost
    "nasa_frost_days"    : get_values("FROST_DAYS"),

    # Cloud
    "cloud_cover"        : get_values("CLOUD_AMT"),
}

df = pd.DataFrame(rows)

# -------------------------------------------------------
# PARSE AND CLEAN
# -------------------------------------------------------

# Parse dates
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# Replace NASA missing value -999 with NaN
df = df.replace(MISSING_VALUE, np.nan)

# Add calendar columns
df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["day"]         = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

# -------------------------------------------------------
# PREVIEW
# -------------------------------------------------------

print(f"\nDataset shape: {df.shape}")
print(f"  Rows:    {len(df):,}")
print(f"  Columns: {len(df.columns)}")

print(f"\nDate range: {df['date'].min().date()} "
      f"to {df['date'].max().date()}")

print(f"\nMissing values per column:")
missing_counts = df.isnull().sum()
missing_pct    = (missing_counts / len(df) * 100).round(1)
for col in df.columns:
    if missing_counts[col] > 0:
        print(f"  {col:<25} {missing_counts[col]:>6} "
              f"({missing_pct[col]:.1f}%)")

print(f"\nNo missing values in: "
      f"{(missing_counts == 0).sum()} columns")

print(f"\nFirst 3 rows (key columns):")
key_cols = [
    "date", "temp_avg", "temp_max", "temp_min",
    "precipitation", "soil_temp_0_10cm",
    "soil_wet_root", "soil_moisture_pctl",
    "gdd_base_10"
]
print(df[key_cols].head(3).to_string())

# -------------------------------------------------------
# QUICK STATISTICS PREVIEW
# -------------------------------------------------------

print(f"\n--- SOIL MOISTURE SUMMARY ---")
print(f"Root zone wetness (0=dry, 1=saturated):")
print(f"  Average: {df['soil_wet_root'].mean():.3f}")
print(f"  Min:     {df['soil_wet_root'].min():.3f}")
print(f"  Max:     {df['soil_wet_root'].max():.3f}")

print(f"\nRoot zone moisture percentile:")
print(f"  Average: {df['soil_moisture_pctl'].mean():.1f}%")
print(f"  Days below 10th percentile (drought): "
      f"{(df['soil_moisture_pctl'] < 10).sum()}")
print(f"  Days below 20th percentile (dry):     "
      f"{(df['soil_moisture_pctl'] < 20).sum()}")

print(f"\n--- GROWING DEGREE DAYS SUMMARY ---")
print(f"Annual GDD accumulation (avg per year):")
n_years = len(df["year"].unique())
for col, label in [
    ("gdd_base_4",  "Base  4°C (onion)"),
    ("gdd_base_7",  "Base  7°C (potato)"),
    ("gdd_base_10", "Base 10°C (tomato, corn)"),
    ("gdd_base_13", "Base 13°C (pepper)"),
]:
    annual_gdd = df.groupby("year")[col].sum().mean()
    print(f"  {label}: {annual_gdd:.0f} GDD/year")

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nData saved to: {OUTPUT_FILE}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\nDone!")
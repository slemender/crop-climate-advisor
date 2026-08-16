# src/nasa_api.py
#
# Fetch 45+ years of daily climate data from NASA POWER
# for Vojvodina, Serbia and save it as a CSV file.
#
# What this script does:
# 1. Asks NASA for daily data from 1981 to July 2026
# 2. Converts the response into a pandas DataFrame
# 3. Displays the first and last few rows
# 4. Saves the data as a CSV file in data/raw/

import requests      # talks to the internet
import pandas as pd  # data analysis library
import numpy as np   # numerical operations
import os            # lets us work with files and folders

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

LATITUDE       = 45.52170251446399
LONGITUDE      = 19.5709129680694
LOCATION_NAME  = "Vojvodina, Serbia"

# Date range - NASA POWER data starts from 1981
# We stop at the last complete month to avoid partial data
START_DATE = "19810101"   # January 1st 1981
END_DATE   = "20260731"   # July 31st 2026 - last complete month

# Where to save our data
OUTPUT_FOLDER = "data/raw"
OUTPUT_FILE   = "data/raw/vojvodina_nasa_power_daily.csv"

# NASA POWER API
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Variables we are requesting
VARIABLES = ",".join([
    "T2M",               # Average temperature at 2 metres (°C)
    "T2M_MAX",           # Maximum daily temperature (°C)
    "T2M_MIN",           # Minimum daily temperature (°C)
    "PRECTOTCORR",       # Total precipitation corrected (mm/day)
    "RH2M",              # Relative humidity at 2 metres (%)
    "ALLSKY_SFC_SW_DWN", # Solar radiation (MJ/m²/day)
    "WS2M",              # Wind speed at 2 metres (m/s)
])

# NASA uses -999 to indicate missing data
MISSING_VALUE = -999.0

# -------------------------------------------------------
# STEP 1 - SEND REQUEST TO NASA
# -------------------------------------------------------

params = {
    "latitude"   : LATITUDE,
    "longitude"  : LONGITUDE,
    "start"      : START_DATE,
    "end"        : END_DATE,
    "community"  : "AG",         # Agricultural dataset
    "parameters" : VARIABLES,
    "format"     : "JSON",
    "header"     : "true",
}

print(f"Requesting NASA POWER data for {LOCATION_NAME}")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Variables: {VARIABLES}")
print("This may take 30-60 seconds for 45 years of data...")
print("-" * 60)

response = requests.get(NASA_POWER_URL, params=params, timeout=120)
# timeout=120 means: if NASA does not respond within 120 seconds, give up

# -------------------------------------------------------
# STEP 2 - CHECK THE RESPONSE
# -------------------------------------------------------

if response.status_code != 200:
    print(f"ERROR - NASA returned status code: {response.status_code}")
    print(response.text[:500])
    exit()

print(f"SUCCESS - NASA responded (status code: {response.status_code})")

# -------------------------------------------------------
# STEP 3 - PARSE THE JSON RESPONSE
# -------------------------------------------------------

data = response.json()

# Check if NASA sent any error messages
if data.get("messages"):
    print(f"NASA messages: {data['messages']}")

# Navigate to the actual parameter data
parameters = data["properties"]["parameter"]

print(f"Variables received: {list(parameters.keys())}")

# -------------------------------------------------------
# STEP 4 - CONVERT TO PANDAS DATAFRAME
# -------------------------------------------------------

# Get all the dates from the temperature data
dates = list(parameters["T2M"].keys())

print(f"Total days received: {len(dates)}")

# Build a dictionary with one entry per variable
rows = {
    "date"          : dates,
    "temp_avg"      : list(parameters["T2M"].values()),
    "temp_max"      : list(parameters["T2M_MAX"].values()),
    "temp_min"      : list(parameters["T2M_MIN"].values()),
    "precipitation" : list(parameters["PRECTOTCORR"].values()),
    "humidity"      : list(parameters["RH2M"].values()),
    "solar_rad"     : list(parameters["ALLSKY_SFC_SW_DWN"].values()),
    "wind_speed"    : list(parameters["WS2M"].values()),
}

# Create the DataFrame
df = pd.DataFrame(rows)

# -------------------------------------------------------
# STEP 5 - CONVERT DATE COLUMN TO PROPER DATE FORMAT
# -------------------------------------------------------

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# -------------------------------------------------------
# STEP 6 - REPLACE MISSING VALUES
# -------------------------------------------------------

# Replace NASA's -999 fill value with NaN (proper missing value)
df = df.replace(MISSING_VALUE, np.nan)

# -------------------------------------------------------
# STEP 7 - ADD USEFUL COLUMNS
# -------------------------------------------------------

df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day

# -------------------------------------------------------
# STEP 8 - DISPLAY A PREVIEW
# -------------------------------------------------------

print("\nFirst 5 rows of data:")
print("-" * 60)
print(df.head())

print("\nLast 5 rows of data:")
print("-" * 60)
print(df.tail())

print("\nDataFrame shape (rows, columns):")
print(df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing value count per column:")
print(df.isnull().sum())

# -------------------------------------------------------
# STEP 9 - SAVE AS CSV
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nData saved to: {OUTPUT_FILE}")
print(f"Total records saved: {len(df)}")
print("\nDone!")
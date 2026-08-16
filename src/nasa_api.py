# src/nasa_api.py
#
# Fetch 40+ years of daily climate data from NASA POWER
# for Vojvodina, Serbia and save it as a CSV file.
#
# What this script does:
# 1. Asks NASA for daily data from 1981 to 2024
# 2. Converts the response into a pandas DataFrame
# 3. Displays the first and last few rows
# 4. Saves the data as a CSV file in data/raw/

import requests   # talks to the internet
import pandas as pd  # data analysis library
import os            # lets us work with files and folders

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

LATITUDE       = 45.52170251446399
LONGITUDE      = 19.5709129680694
LOCATION_NAME  = "Vojvodina, Serbia"

# Date range - NASA POWER data starts from 1981
START_DATE = "19810101"   # January 1st 1981
END_DATE   = "20241231"   # December 31st 2024

# Where to save our data
OUTPUT_FOLDER = "data/raw"
OUTPUT_FILE   = "data/raw/vojvodina_nasa_power_daily.csv"

# NASA POWER API
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Variables we are requesting
# We are requesting more variables this time
VARIABLES = ",".join([
    "T2M",          # Average temperature at 2 metres (°C)
    "T2M_MAX",      # Maximum daily temperature (°C)
    "T2M_MIN",      # Minimum daily temperature (°C)
    "PRECTOTCORR",  # Total precipitation corrected (mm/day)
    "RH2M",         # Relative humidity at 2 metres (%)
    "ALLSKY_SFC_SW_DWN",  # Solar radiation (MJ/m²/day)
    "WS2M",         # Wind speed at 2 metres (m/s)
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
print("This may take 30-60 seconds for 40 years of data...")
print("-" * 60)

response = requests.get(NASA_POWER_URL, params=params, timeout=120)
# timeout=120 means: if NASA doesn't respond within 120 seconds, give up

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

# First, let's get all the dates from the temperature data
# dates look like "20240101"
dates = list(parameters["T2M"].keys())

print(f"Total days received: {len(dates)}")

# Build a dictionary with one entry per variable
# Each entry is a list of values, one per date
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
# Think of a DataFrame as a spreadsheet with named columns
df = pd.DataFrame(rows)

# -------------------------------------------------------
# STEP 5 - CONVERT DATE COLUMN TO PROPER DATE FORMAT
# -------------------------------------------------------

# Right now dates look like "20240101" (a string)
# We want pandas to understand these as real dates
# so we can filter by year, month, day later
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# -------------------------------------------------------
# STEP 6 - REPLACE MISSING VALUES
# -------------------------------------------------------

# NASA uses -999 for missing data
# We replace -999 with NaN (Not a Number) - 
# pandas understands NaN as "missing" and handles it properly
import numpy as np

df = df.replace(MISSING_VALUE, np.nan)

# -------------------------------------------------------
# STEP 7 - ADD USEFUL COLUMNS
# -------------------------------------------------------

# Add year, month, day as separate columns
# This makes filtering much easier later
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day

# -------------------------------------------------------
# STEP 8 - DISPLAY A PREVIEW
# -------------------------------------------------------

print("\nFirst 5 rows of data:")
print("-" * 60)
print(df.head())   # head() shows the first 5 rows

print("\nLast 5 rows of data:")
print("-" * 60)
print(df.tail())   # tail() shows the last 5 rows

print("\nDataFrame shape (rows, columns):")
print(df.shape)    # tells us how many rows and columns we have

print("\nColumn data types:")
print(df.dtypes)   # tells us what type each column is

print("\nMissing value count per column:")
print(df.isnull().sum())  # counts missing values in each column

# -------------------------------------------------------
# STEP 9 - SAVE AS CSV
# -------------------------------------------------------

# Make sure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)
# index=False means don't save the row numbers as a column

print(f"\nData saved to: {OUTPUT_FILE}")
print(f"Total records saved: {len(df)}")
print("\nDone!")
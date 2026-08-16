# src/config.py
#
# Central configuration loader.
#
# Reads all settings from the .env file in the project root.
# Every other script imports from here — nothing reads .env
# directly and nothing has hardcoded coordinates.
#
# To change location: edit .env only.
# No Python files need to be touched.
#
# Usage in other scripts:
#   from config import CONFIG
#   lat = CONFIG["latitude"]
#   lon = CONFIG["longitude"]

import os
from pathlib import Path
from dotenv import load_dotenv


# -------------------------------------------------------
# FIND AND LOAD .env FILE
# -------------------------------------------------------
# Walk up from this file's location to find .env
# This works regardless of where the script is run from.

def _find_env_file():
    """
    Search for .env file starting from this script's
    directory and walking up to the project root.
    """
    # Start from the src/ directory
    current = Path(__file__).parent

    # Check current directory and up to 3 levels above
    for _ in range(4):
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        current = current.parent

    return None


_env_path = _find_env_file()

if _env_path is None:
    raise FileNotFoundError(
        "Could not find .env file.\n"
        "Expected at project root: .env\n"
        "Please create it with:\n"
        "  LATITUDE=your_latitude\n"
        "  LONGITUDE=your_longitude\n"
        "  LOCATION_NAME=Your Location"
    )

# Load the .env file into environment variables
load_dotenv(_env_path)

# -------------------------------------------------------
# READ AND VALIDATE REQUIRED VARIABLES
# -------------------------------------------------------

def _require(key):
    """
    Read a required environment variable.
    Raises a clear error if it is missing.
    """
    val = os.getenv(key)
    if val is None or val.strip() == "":
        raise ValueError(
            f"Missing required variable in .env: {key}\n"
            f"Please add it to your .env file:\n"
            f"  {key}=your_value_here"
        )
    return val.strip()


def _optional(key, default=None):
    """
    Read an optional environment variable.
    Returns default if not set.
    """
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip()


# -------------------------------------------------------
# BUILD CONFIG DICTIONARY
# -------------------------------------------------------

try:
    _lat = float(_require("LATITUDE"))
    _lon = float(_require("LONGITUDE"))
except ValueError as e:
    raise ValueError(
        f"LATITUDE and LONGITUDE must be valid numbers.\n"
        f"Error: {e}"
    )

CONFIG = {
    # Location
    "latitude"      : _lat,
    "longitude"     : _lon,
    "location_name" : _optional("LOCATION_NAME", "Unknown Location"),

    # File paths (relative to project root)
    "raw_data_file"   : "data/raw/nasa_power_daily.csv",
    "clean_data_file" : "data/processed/vojvodina_clean.csv",
    "crops_file"      : "data/crops.json",
    "risk_scores_file": "data/processed/planting_date_risk_scores.csv",
    "ml_pred_file"    : "data/processed/ml_planting_predictions.csv",

    # NASA POWER API settings
    "nasa_power_url"  : "https://power.larc.nasa.gov/api/temporal/daily/point",
    "nasa_community"  : "AG",
    "nasa_start_date" : _optional("NASA_START_DATE", "20000101"),
    "nasa_end_date"   : _optional("NASA_END_DATE",   "20260731"),

    # Analysis settings
    "analysis_start"  : int(_optional("ANALYSIS_START", "2000")),
    "analysis_end"    : int(_optional("ANALYSIS_END",   "2026")),

    # Environment file location (for reference)
    "env_file"        : str(_env_path),

    # Copernicus Climate Data Store (ERA5-Land soil temperature)
    "cds_api_key"     : _optional("CDS_API_KEY", None),
    "era5_start_year" : int(_optional("ERA5_START_YEAR", "1950")),
    "era5_end_year"   : int(_optional("ERA5_END_YEAR",   "2025")),
    "soil_temp_file"  : "data/raw/era5_soil_temperature.csv",
    "soil_temp_clean" : "data/processed/soil_temperature_clean.csv",
}

# -------------------------------------------------------
# MAIN - runs when executed directly
# -------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Loaded from: {CONFIG['env_file']}")
    print()
    print(f"Location:    {CONFIG['location_name']}")
    print(f"Latitude:    {CONFIG['latitude']}")
    print(f"Longitude:   {CONFIG['longitude']}")
    print()
    print(f"Data range:  {CONFIG['nasa_start_date']} "
          f"to {CONFIG['nasa_end_date']}")
    print(f"Analysis:    {CONFIG['analysis_start']} "
          f"to {CONFIG['analysis_end']}")
    print()
    print("File paths:")
    for key in ["raw_data_file", "clean_data_file",
                "crops_file", "risk_scores_file"]:
        print(f"  {key:<22}: {CONFIG[key]}")
    print()
    print("=== CONFIG LOADED SUCCESSFULLY ===")
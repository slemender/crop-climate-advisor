# src/crop_model.py
#
# Crop Science Database — loads from data/crops.json
#
# To add a new crop: edit data/crops.json
# No Python changes needed.
#
# The JSON file contains all crop requirements including:
# - Temperature thresholds (frost, heat, optimal)
# - Water requirements by month
# - GDD requirements
# - Growing season length
# - Traditional planting windows
# - Scientific sources

import json
import os

# -------------------------------------------------------
# LOAD CROPS FROM JSON
# -------------------------------------------------------

# Find the crops.json file relative to this script
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT    = os.path.dirname(_THIS_DIR)
_CROPS_FILE = os.path.join(_PROJECT, "data", "crops.json")

def _load_crops():
    """
    Load crop data from the JSON file.
    Called once when this module is first imported.
    Raises a clear error if the file is missing.
    """
    if not os.path.exists(_CROPS_FILE):
        raise FileNotFoundError(
            f"Crop data file not found: {_CROPS_FILE}\n"
            f"Expected at: data/crops.json\n"
            f"Please ensure the file exists in your project."
        )

    with open(_CROPS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert monthly water keys from strings to integers
    # JSON requires string keys but we want integer month numbers
    for crop_key, crop in data.items():
        if "water" in crop:
            water = crop["water"]

            if "monthly_water_need_mm" in water:
                water["monthly_water_need_mm"] = {
                    int(k): v
                    for k, v in water["monthly_water_need_mm"].items()
                }

            if "critical_month_min_rain_mm" in water:
                water["critical_month_min_rain_mm"] = {
                    int(k): v
                    for k, v in
                    water["critical_month_min_rain_mm"].items()
                }

    return data


# Load crops once when module is imported
CROPS = _load_crops()

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def get_crop(crop_name):
    """
    Return the full data dictionary for a crop.

    Usage:
        potato = get_crop("potato")
        print(potato["frost"]["foliage_kill_c"])

    Raises ValueError if crop is not found.
    """
    crop_name = crop_name.lower().strip()
    if crop_name not in CROPS:
        available = list(CROPS.keys())
        raise ValueError(
            f"Crop '{crop_name}' not found.\n"
            f"Available crops: {available}\n"
            f"To add a crop, edit: data/crops.json"
        )
    return CROPS[crop_name]


def get_frost_kill_temp(crop_name):
    """
    Return the temperature at which this crop is killed
    by frost. Returns a negative number (degrees Celsius).
    """
    crop = get_crop(crop_name)
    return crop["frost"]["foliage_kill_c"]


def get_heat_stress_temp(crop_name):
    """
    Return the temperature at which this crop experiences
    significant heat stress (severe_stress_c).
    """
    crop = get_crop(crop_name)
    heat = crop["heat"]

    # Different crops may use different field names
    # Try each in order of preference
    for key in ["severe_stress_c",
                "flower_drop_severe_c",
                "pollen_damage_c",
                "quality_decline_c"]:
        if key in heat:
            return heat[key]

    return 35.0  # safe default


def get_gdd_base(crop_name):
    """Return the base temperature for GDD calculation."""
    return get_crop(crop_name)["gdd"]["base_temp_c"]


def get_gdd_to_maturity(crop_name, variety="medium"):
    """
    Return approximate GDD needed to reach maturity.
    variety: "early", "medium", or "late"
    """
    gdd = get_crop(crop_name)["gdd"]
    if variety == "early":
        return gdd["maturity_gdd_min"]
    elif variety == "late":
        return gdd["maturity_gdd_max"]
    else:
        return (gdd["maturity_gdd_min"] + gdd["maturity_gdd_max"]) / 2


def get_monthly_water_need(crop_name, month):
    """
    Return the crop water requirement for a given month in mm.
    Returns 0 if crop is not actively growing that month.
    """
    crop = get_crop(crop_name)
    return crop["water"]["monthly_water_need_mm"].get(month, 0)


def get_critical_rain_threshold(crop_name, month):
    """
    Return the minimum rainfall needed in a given month
    to avoid significant yield loss.
    Returns None if this month is not a critical period.
    """
    crop     = get_crop(crop_name)
    critical = crop["water"].get("critical_month_min_rain_mm", {})
    return critical.get(month, None)


def get_rainfall_deficit(crop_name, month, actual_rainfall_mm):
    """
    Calculate rainfall deficit for a crop in a given month.
    Positive = deficit (need more rain)
    Negative = surplus (more rain than needed)
    """
    needed = get_monthly_water_need(crop_name, month)
    if needed == 0:
        return 0
    return needed - actual_rainfall_mm


def list_crops():
    """Print a summary of all crops in the database."""
    print(f"\n=== CROP DATABASE ===")
    print(f"Loaded from: {_CROPS_FILE}")
    print(f"Total crops: {len(CROPS)}\n")

    for key, crop in CROPS.items():
        print(f"{crop['name']} ({crop['latin']})")
        print(f"  Season type:       {crop['season_type']}")
        print(f"  Frost kills at:    "
              f"{crop['frost']['foliage_kill_c']}°C")
        print(f"  Heat stress at:    {get_heat_stress_temp(key)}°C")
        print(f"  GDD base temp:     "
              f"{crop['gdd']['base_temp_c']}°C")
        print(f"  GDD to maturity:   "
              f"{crop['gdd']['maturity_gdd_min']}"
              f"–{crop['gdd']['maturity_gdd_max']} GDD")
        print(f"  Total water need:  "
              f"{crop['water']['total_season_mm']}mm")
        print(f"  Drought risk:      "
              f"{crop['water']['drought_sensitivity']}")
        print()


# -------------------------------------------------------
# MAIN - runs when executed directly
# -------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CROP DATABASE")
    print(f"Source: {_CROPS_FILE}")
    print("=" * 60)

    list_crops()

    print("=== KEY THRESHOLDS ===\n")
    print(f"{'Crop':<14} {'Frost Kill':>12} "
          f"{'Heat Stress':>12} {'GDD Base':>10} "
          f"{'Water(mm)':>11}")
    print("-" * 62)

    for key, crop in CROPS.items():
        print(f"{crop['name']:<14} "
              f"{crop['frost']['foliage_kill_c']:>10.1f}°C "
              f"{get_heat_stress_temp(key):>10.1f}°C "
              f"{crop['gdd']['base_temp_c']:>9.0f}°C "
              f"{crop['water']['total_season_mm']:>10}mm")

    print(f"\nTo add a new crop: edit data/crops.json")
    print(f"No Python changes needed.")
    print("\n=== CROP DATABASE LOADED SUCCESSFULLY ===")
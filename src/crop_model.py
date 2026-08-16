# src/crop_model.py
#
# Crop Science Database for Vojvodina, Serbia
#
# IMPORTANT: This version includes detailed rainfall
# requirements because the grower has NO IRRIGATION.
# Rainfall timing and amount are as important as
# temperature for crop success in this system.
#
# Data sources:
# - FAO Crop Water Requirements (Doorenbos & Pruitt, 1977)
# - FAO Irrigation and Drainage Paper 33 and 56
# - FAO Agronomy Series publications
# - University extension services (Cornell, Wageningen)
# - USDA Agricultural Research Service publications
# - European Commission JRC MARS crop monitoring
#
# All numbers are approximate and vary by:
# - cultivar and variety
# - soil type and water holding capacity
# - local microclimate
# - planting method

# -------------------------------------------------------
# CROP DATABASE
# -------------------------------------------------------

CROPS = {

    # ===================================================
    # POTATO (Solanum tuberosum)
    # ===================================================
    # Sources: FAO Production Yearbook, EAPR guidelines,
    #          FAO Irrigation Paper 33, Potato Association
    #          of America agronomy papers
    # ===================================================

    "potato": {

        "name"        : "Potato",
        "latin"       : "Solanum tuberosum",
        "season_type" : "cool",
        "notes": (
            "Cool-season crop. Heat is the primary threat. "
            "Without irrigation, rainfall during tuber bulking "
            "is critical. Early varieties that mature before "
            "July avoid both heat stress and summer drought. "
            "This is strongly recommended for Vojvodina "
            "given recent climate trends."
        ),

        # --- Germination / Sprouting ---
        "germination": {
            "soil_temp_min_c"     : 7,
            "soil_temp_optimal_c" : 15,
            "soil_temp_max_c"     : 29,
            "note": (
                "Soil temperature at 10cm depth. "
                "Air temperature alone is insufficient — "
                "soil warms more slowly than air in spring."
            ),
        },

        # --- Frost Sensitivity ---
        "frost": {
            "foliage_damage_c" : -1.0,
            "foliage_kill_c"   : -3.0,
            "tuber_damage_c"   : -1.5,
            "tolerance_stage"  : "none",
            "note": (
                "Emerging shoots are very frost sensitive. "
                "Tubers can tolerate brief light frost if "
                "well covered with soil. A late frost in "
                "April can kill emerged shoots but plants "
                "often recover from the tuber."
            ),
        },

        # --- Heat Sensitivity ---
        "heat": {
            "growth_slows_c"           : 25,
            "severe_stress_c"          : 35,
            "tuber_initiation_stops_c" : 29,
            "note": (
                "Tuber initiation stops above 29°C night temperature. "
                "Given Vojvodina now averages 20+ days above 35°C "
                "in summer, late-season potatoes face serious heat risk. "
                "Early varieties planted March-April and harvested "
                "before July are strongly recommended."
            ),
        },

        # --- Preferred Growing Conditions ---
        "optimal": {
            "day_temp_c"   : 18,
            "night_temp_c" : 12,
            "temp_range_c" : (15, 20),
            "note": (
                "Potatoes prefer cool nights. The day-night "
                "temperature difference is important for "
                "tuber quality and starch accumulation."
            ),
        },

        # --- Water Requirements (NO IRRIGATION) ---
        # Source: FAO Irrigation and Drainage Paper 33 and 56
        # Kc values (crop coefficients) from FAO-56
        "water": {
            "total_season_mm"     : 500,
            "critical_period"     : "tuber initiation to bulking",
            "drought_sensitivity" : "high",
            "irrigation_benefit"  : "very high",

            # Monthly water needs by growth stage
            # These are CROP water requirements (ETc)
            # not just rainfall — they account for
            # how much water the plant actually uses
            "monthly_water_need_mm": {
                # month: approximate crop water need (mm)
                # Based on FAO-56 Kc values and
                # reference ET for Vojvodina region
                3  : 40,   # March     - emergence, low need
                4  : 75,   # April     - vegetative growth
                5  : 110,  # May       - rapid growth
                6  : 130,  # June      - tuber initiation (CRITICAL)
                7  : 120,  # July      - tuber bulking (CRITICAL)
                8  : 80,   # August    - maturation
                9  : 40,   # September - late/harvest
            },

            # Minimum rainfall needed in critical months
            # Below this, yield loss is likely without irrigation
            "critical_month_min_rain_mm": {
                6: 80,    # June  - need at least 80mm
                7: 80,    # July  - need at least 80mm
                8: 60,    # August
            },

            # Maximum consecutive dry days tolerated
            # per growth stage before stress begins
            "max_dry_days": {
                "emergence"           : 14,
                "vegetative"          : 10,
                "tuber_initiation"    : 7,   # most sensitive
                "tuber_bulking"       : 7,   # most sensitive
                "maturation"          : 14,
            },

            # Vojvodina context
            "no_irrigation_risk": (
                "HIGH RISK without irrigation. "
                "July average rainfall (57mm) is well below "
                "the 120mm crop water need in that month. "
                "Early varieties harvested by late June "
                "dramatically reduce this risk."
            ),
        },

        # --- Growing Degree Days ---
        "gdd": {
            "base_temp_c"      : 7,
            "maturity_gdd_min" : 1000,
            "maturity_gdd_max" : 1800,
            "note": (
                "Early varieties: ~1000-1200 GDD. "
                "Medium varieties: ~1300-1500 GDD. "
                "Late varieties: ~1600-1800 GDD."
            ),
        },

        # --- Days to Maturity ---
        "maturity": {
            "early_days"  : 70,
            "medium_days" : 90,
            "late_days"   : 120,
        },

        # --- Traditional Vojvodina Planting Window ---
        "traditional_planting": {
            "earliest"      : "March 15",
            "typical_start" : "April 1",
            "typical_end"   : "April 20",
            "latest"        : "May 1",
            "note": (
                "For no-irrigation growing, early varieties "
                "planted March-April and harvested June-July "
                "are strongly preferred. This avoids both "
                "peak heat and the summer rainfall deficit."
            ),
        },
    },

    # ===================================================
    # TOMATO (Solanum lycopersicum)
    # ===================================================
    # Sources: FAO, UC Davis Vegetable Research Center,
    #          Cornell Cooperative Extension
    # ===================================================

    "tomato": {

        "name"        : "Tomato",
        "latin"       : "Solanum lycopersicum",
        "season_type" : "warm",
        "notes": (
            "Warm-season crop. Cannot tolerate any frost. "
            "Without irrigation, tomatoes are highly vulnerable "
            "to Vojvodina's summer rainfall deficit. "
            "July-August is both the hottest AND the driest "
            "period — exactly when tomatoes need water most "
            "for fruit development. Rainfall prediction is "
            "critical for this crop."
        ),

        # --- Germination ---
        "germination": {
            "soil_temp_min_c"     : 10,
            "soil_temp_optimal_c" : 22,
            "soil_temp_max_c"     : 35,
            "note": (
                "Typically started indoors 6-8 weeks before "
                "transplanting. Do not transplant until frost "
                "risk is very low and nights are above 10°C."
            ),
        },

        # --- Frost Sensitivity ---
        "frost": {
            "foliage_damage_c" : -0.5,
            "foliage_kill_c"   : -1.5,
            "tolerance_stage"  : "none",
            "note": (
                "Tomatoes are killed by even light frost. "
                "Do not transplant until the 90th percentile "
                "last frost date has passed — approximately "
                "April 24 in Vojvodina."
            ),
        },

        # --- Heat Sensitivity ---
        "heat": {
            "flower_drop_starts_c" : 32,
            "flower_drop_severe_c" : 35,
            "severe_stress_c"      : 35,
            "pollen_damage_c"      : 38,
            "note": (
                "With 20+ days above 35°C per year in recent "
                "years, fruit set failure in July-August is "
                "a serious risk. Heat-tolerant varieties are "
                "increasingly important in Vojvodina."
            ),
        },

        # --- Preferred Growing Conditions ---
        "optimal": {
            "day_temp_c"   : 24,
            "night_temp_c" : 16,
            "temp_range_c" : (18, 27),
            "note": (
                "Night temperatures below 10°C slow growth. "
                "Optimal window is May through early July "
                "before peak summer heat."
            ),
        },

        # --- Water Requirements (NO IRRIGATION) ---
        "water": {
            "total_season_mm"     : 600,
            "critical_period"     : "flowering and fruit set",
            "drought_sensitivity" : "high",
            "irrigation_benefit"  : "very high",

            "monthly_water_need_mm": {
                4  : 50,   # April     - transplant establishment
                5  : 100,  # May       - vegetative growth
                6  : 140,  # June      - flowering begins (CRITICAL)
                7  : 160,  # July      - peak fruit set (MOST CRITICAL)
                8  : 140,  # August    - fruit development (CRITICAL)
                9  : 80,   # September - late harvest
            },

            "critical_month_min_rain_mm": {
                6: 100,   # June  - flowering
                7: 110,   # July  - fruit set (highest need)
                8: 100,   # August - fruit development
            },

            "max_dry_days": {
                "transplant"       : 7,
                "vegetative"       : 10,
                "flowering"        : 5,   # most sensitive stage
                "fruit_set"        : 5,   # most sensitive stage
                "fruit_development": 7,
                "maturation"       : 10,
            },

            "no_irrigation_risk": (
                "VERY HIGH RISK without irrigation. "
                "Tomatoes need ~160mm in July but Vojvodina "
                "averages only 57mm. The rainfall deficit in "
                "July alone is ~103mm. Without irrigation, "
                "expect significant yield loss in dry years. "
                "Rainfall prediction is essential for planning."
            ),
        },

        # --- Growing Degree Days ---
        "gdd": {
            "base_temp_c"      : 10,
            "maturity_gdd_min" : 1000,
            "maturity_gdd_max" : 1400,
            "note": (
                "Early cherry/salad types: ~1000 GDD. "
                "Main crop beefsteak types: ~1200-1400 GDD."
            ),
        },

        # --- Days to Maturity ---
        "maturity": {
            "early_days"  : 60,
            "medium_days" : 75,
            "late_days"   : 95,
        },

        # --- Traditional Vojvodina Planting Window ---
        "traditional_planting": {
            "start_indoors"   : "February 15 - March 1",
            "transplant_safe" : "May 1",
            "transplant_late" : "May 20",
            "note": (
                "Without irrigation, earlier planting is "
                "actually beneficial — it allows fruit to "
                "develop in June when rainfall is higher "
                "before the July-August drought risk peaks."
            ),
        },
    },

    # ===================================================
    # ONION (Allium cepa)
    # ===================================================
    # Sources: FAO, Wageningen University onion production
    #          guidelines, USDA Vegetable Production Handbook
    # ===================================================

    "onion": {

        "name"        : "Onion",
        "latin"       : "Allium cepa",
        "season_type" : "cool",
        "notes": (
            "Cool-season crop with good frost tolerance. "
            "Without irrigation, the main risk is rainfall "
            "during bulb development in June-July. "
            "Onions have shallow roots and cannot access "
            "deep soil moisture — making them vulnerable "
            "to even short dry spells during bulbing."
        ),

        # --- Germination ---
        "germination": {
            "soil_temp_min_c"     : 2,
            "soil_temp_optimal_c" : 20,
            "soil_temp_max_c"     : 35,
            "note": (
                "Onions can be direct seeded very early. "
                "Commonly started indoors January-February "
                "or direct seeded in March."
            ),
        },

        # --- Frost Sensitivity ---
        "frost": {
            "foliage_damage_c" : -3.0,
            "foliage_kill_c"   : -8.0,
            "tolerance_stage"  : "established plant",
            "note": (
                "Onions are significantly more frost tolerant "
                "than tomatoes or cucumbers. Established plants "
                "survive to approximately -6°C to -8°C. "
                "Young seedlings are more vulnerable (~-3°C)."
            ),
        },

        # --- Heat Sensitivity ---
        "heat": {
            "quality_decline_c" : 30,
            "severe_stress_c"   : 35,
            "note": (
                "Extreme heat causes premature bolting and "
                "poor bulb quality. July-August heat is a "
                "risk for late-season onions."
            ),
        },

        # --- Preferred Growing Conditions ---
        "optimal": {
            "day_temp_c"   : 20,
            "night_temp_c" : 10,
            "temp_range_c" : (13, 24),
            "note": (
                "Long-day varieties required at latitude 45°N. "
                "Bulbing triggered by >14-16 hours daylight "
                "occurring naturally in May-June in Vojvodina."
            ),
        },

        # --- Water Requirements (NO IRRIGATION) ---
        "water": {
            "total_season_mm"     : 450,
            "critical_period"     : "bulb development",
            "drought_sensitivity" : "moderate-high",
            "irrigation_benefit"  : "high",

            "monthly_water_need_mm": {
                3  : 35,   # March     - early establishment
                4  : 60,   # April     - vegetative growth
                5  : 90,   # May       - rapid leaf growth
                6  : 110,  # June      - bulb initiation (CRITICAL)
                7  : 90,   # July      - bulb development (CRITICAL)
                8  : 40,   # August    - maturation/harvest
                            # reduce water for storage quality
            },

            "critical_month_min_rain_mm": {
                5: 60,    # May  - leaf growth
                6: 75,    # June - bulb initiation
                7: 60,    # July - bulb development
            },

            "max_dry_days": {
                "seedling"         : 7,
                "vegetative"       : 10,
                "bulb_initiation"  : 7,   # most sensitive
                "bulb_development" : 10,
                "maturation"       : 14,  # dry conditions good here
            },

            "no_irrigation_risk": (
                "MODERATE-HIGH RISK without irrigation. "
                "Onions have very shallow roots (~30cm) "
                "and cannot tolerate even short dry spells "
                "during bulbing. June rainfall (75mm avg) "
                "is close to the 110mm need but variable. "
                "Dry June years will significantly reduce yield. "
                "Early planting to complete bulbing in June "
                "before July rainfall decline is recommended."
            ),
        },

        # --- Growing Degree Days ---
        "gdd": {
            "base_temp_c"      : 4,
            "maturity_gdd_min" : 1200,
            "maturity_gdd_max" : 1800,
            "note": (
                "Day length is as important as temperature "
                "for bulbing initiation. Use long-day varieties."
            ),
        },

        # --- Days to Maturity ---
        "maturity": {
            "from_seed_days"       : 150,
            "from_transplant_days" : 120,
            "from_sets_days"       : 90,
        },

        # --- Traditional Vojvodina Planting Window ---
        "traditional_planting": {
            "sets_earliest"       : "February 15",
            "sets_typical"        : "March 1 - March 20",
            "transplants_typical" : "March 10 - April 1",
            "direct_seed_typical" : "March 1 - April 1",
            "note": (
                "Early planting is key without irrigation. "
                "Getting bulbs to develop in May-June rather "
                "than July-August uses the better rainfall "
                "period and avoids summer heat."
            ),
        },
    },

    # ===================================================
    # CUCUMBER (Cucumis sativus)
    # ===================================================
    # Sources: FAO, UC Davis Vegetable Research Center,
    #          Cornell Cooperative Extension cucumber guide
    # ===================================================

    "cucumber": {

        "name"        : "Cucumber",
        "latin"       : "Cucumis sativus",
        "season_type" : "warm",
        "notes": (
            "Warm-season crop. Extremely frost sensitive. "
            "Cucumbers are approximately 95% water by weight — "
            "making consistent rainfall absolutely critical. "
            "Without irrigation, cucumber success depends "
            "heavily on July-August rainfall, which is exactly "
            "when Vojvodina is at its driest. "
            "Rainfall prediction is the most important factor "
            "for this crop after frost-free dates."
        ),

        # --- Germination ---
        "germination": {
            "soil_temp_min_c"     : 15,
            "soil_temp_optimal_c" : 30,
            "soil_temp_max_c"     : 40,
            "note": (
                "Cucumbers need warm soil. Planting in cold "
                "soil causes poor germination and rotting seeds. "
                "Soil should be consistently above 15°C."
            ),
        },

        # --- Frost Sensitivity ---
        "frost": {
            "foliage_damage_c" : -0.5,
            "foliage_kill_c"   : -1.0,
            "tolerance_stage"  : "none",
            "note": (
                "Extremely frost sensitive. Even a very light "
                "frost will damage or kill cucumber plants "
                "at any growth stage. Do not plant until "
                "all frost risk has clearly passed."
            ),
        },

        # --- Heat Sensitivity ---
        "heat": {
            "quality_decline_c" : 35,
            "severe_stress_c"   : 40,
            "note": (
                "Cucumbers are more heat tolerant than tomatoes "
                "during flowering, but fruit quality declines "
                "above 35°C — bitterness increases significantly. "
                "Water stress combined with heat is especially "
                "damaging — a common combination in Vojvodina summers."
            ),
        },

        # --- Preferred Growing Conditions ---
        "optimal": {
            "day_temp_c"   : 27,
            "night_temp_c" : 18,
            "temp_range_c" : (24, 32),
            "note": (
                "Most heat-loving of our four crops. "
                "Main risks are frost at planting time and "
                "drought combined with heat in July-August."
            ),
        },

        # --- Water Requirements (NO IRRIGATION) ---
        "water": {
            "total_season_mm"     : 500,
            "critical_period"     : "flowering and fruit development",
            "drought_sensitivity" : "very high",
            "irrigation_benefit"  : "very high",

            "monthly_water_need_mm": {
                5  : 80,   # May       - establishment
                6  : 130,  # June      - vegetative and flowering
                7  : 160,  # July      - peak fruiting (MOST CRITICAL)
                8  : 140,  # August    - continued fruiting (CRITICAL)
                9  : 60,   # September - late harvest
            },

            "critical_month_min_rain_mm": {
                6: 90,    # June  - flowering
                7: 110,   # July  - peak fruiting (highest need)
                8: 100,   # August - continued fruiting
            },

            "max_dry_days": {
                "establishment"    : 7,
                "vegetative"       : 7,
                "flowering"        : 4,   # extremely sensitive
                "fruit_set"        : 4,   # extremely sensitive
                "fruit_development": 5,
                "maturation"       : 7,
            },

            "no_irrigation_risk": (
                "VERY HIGH RISK without irrigation. "
                "Cucumbers need ~160mm in July but Vojvodina "
                "averages only 57mm — a deficit of ~103mm. "
                "Water stress causes bitter fruit, poor set, "
                "and plant wilting. This crop is the most "
                "rainfall-dependent of our four crops. "
                "Rainfall prediction for July-August is essential. "
                "Consider second sowing in July if June-July "
                "rainfall forecast is favorable."
            ),
        },

        # --- Growing Degree Days ---
        "gdd": {
            "base_temp_c"      : 10,
            "maturity_gdd_min" : 700,
            "maturity_gdd_max" : 1000,
            "note": (
                "Fast maturing relative to other crops. "
                "Can be planted late April and harvested "
                "before worst July-August heat and drought."
            ),
        },

        # --- Days to Maturity ---
        "maturity": {
            "early_days"  : 50,
            "medium_days" : 60,
            "late_days"   : 70,
        },

        # --- Traditional Vojvodina Planting Window ---
        "traditional_planting": {
            "earliest_safe" : "April 25",
            "typical_start" : "May 1",
            "typical_end"   : "May 20",
            "second_sowing" : "July 1",
            "note": (
                "Without irrigation, early planting (late April) "
                "is strongly recommended so fruit develops in "
                "June-early July when rainfall is higher. "
                "A second sowing in July is only advisable "
                "if July-August rainfall is forecast to be "
                "above average."
            ),
        },
    },
}

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def get_crop(crop_name):
    """
    Return the full data dictionary for a crop.

    Usage:
        potato = get_crop("potato")
        print(potato["frost"]["foliage_kill_c"])
    """
    crop_name = crop_name.lower().strip()
    if crop_name not in CROPS:
        available = list(CROPS.keys())
        raise ValueError(
            f"Crop '{crop_name}' not found. "
            f"Available crops: {available}"
        )
    return CROPS[crop_name]


def get_frost_kill_temp(crop_name):
    """
    Return the temperature at which this crop is killed by frost.
    Returns a negative number (degrees Celsius).
    """
    crop = get_crop(crop_name)
    return crop["frost"]["foliage_kill_c"]


def get_heat_stress_temp(crop_name):
    """
    Return the temperature at which this crop experiences
    significant heat stress.
    """
    crop = get_crop(crop_name)
    heat = crop["heat"]
    if "severe_stress_c" in heat:
        return heat["severe_stress_c"]
    elif "flower_drop_severe_c" in heat:
        return heat["flower_drop_severe_c"]
    else:
        return 35.0


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
    Return the crop water requirement for a given month.
    Returns mm of water needed. Returns 0 if crop is not
    actively growing in that month.
    """
    crop          = get_crop(crop_name)
    monthly_needs = crop["water"]["monthly_water_need_mm"]
    return monthly_needs.get(month, 0)


def get_rainfall_deficit(crop_name, month, actual_rainfall_mm):
    """
    Calculate how much rainfall deficit exists for a crop
    in a given month.

    Returns:
        positive number = deficit (need more rain)
        negative number = surplus (more rain than needed)
        0 = balanced
    """
    needed = get_monthly_water_need(crop_name, month)
    if needed == 0:
        return 0  # crop not growing in this month
    deficit = needed - actual_rainfall_mm
    return deficit


def get_critical_rain_threshold(crop_name, month):
    """
    Return the minimum rainfall needed in a given month
    to avoid significant yield loss.
    Returns None if this month is not a critical period.
    """
    crop = get_crop(crop_name)
    critical = crop["water"].get("critical_month_min_rain_mm", {})
    return critical.get(month, None)


def list_crops():
    """Print a summary of all crops in the database."""
    print("\n=== CROP DATABASE SUMMARY ===\n")
    for key, crop in CROPS.items():
        print(f"{crop['name']} ({crop['latin']})")
        print(f"  Season type:       {crop['season_type']}")
        print(f"  Frost kills at:    {crop['frost']['foliage_kill_c']}°C")
        print(f"  Heat stress at:    {get_heat_stress_temp(key)}°C")
        print(f"  GDD base temp:     {crop['gdd']['base_temp_c']}°C")
        print(f"  GDD to maturity:   "
              f"{crop['gdd']['maturity_gdd_min']}"
              f"–{crop['gdd']['maturity_gdd_max']} GDD")
        print(f"  Total water need:  {crop['water']['total_season_mm']}mm")
        print(f"  Drought risk:      {crop['water']['drought_sensitivity']}")
        print(f"  No-irrigation risk:{crop['water']['no_irrigation_risk'][:60]}...")
        print()


def show_rainfall_requirements():
    """
    Print a detailed monthly rainfall requirements table
    for all crops. Critical for no-irrigation planning.
    """
    print("\n=== MONTHLY WATER REQUIREMENTS (No Irrigation) ===\n")
    print("Compared against Vojvodina historical average rainfall.\n")

    # Historical average rainfall from our NASA data
    vojvodina_avg_rain = {
        1: 42,  2: 38,  3: 44,  4: 49,
        5: 69,  6: 75,  7: 57,  8: 51,
        9: 56, 10: 52, 11: 54, 12: 50,
    }

    month_names = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    # Header
    print(f"{'Month':<8}", end="")
    print(f"{'Rain(avg)':>10}", end="")
    for key in CROPS:
        name = CROPS[key]["name"]
        print(f"{name:>12}", end="")
    print()

    print(f"{'':8}{'mm/month':>10}", end="")
    for key in CROPS:
        print(f"{'need(mm)':>12}", end="")
    print()
    print("-" * (18 + 12 * len(CROPS)))

    for month in range(1, 13):
        avg_rain = vojvodina_avg_rain[month]
        print(f"{month_names[month-1]:<8}{avg_rain:>10}", end="")

        for key in CROPS:
            need    = get_monthly_water_need(key, month)
            deficit = need - avg_rain if need > 0 else 0

            if need == 0:
                # Crop not growing this month
                print(f"{'—':>12}", end="")
            elif deficit > 30:
                # Significant deficit
                print(f"{need:>9}({'DEFICIT':>0})", end="")
            elif deficit > 0:
                # Minor deficit
                print(f"{need:>9}({'low':>0})", end="")
            else:
                # Surplus - rainfall adequate
                print(f"{need:>12}", end="")
        print()

    print("\nDEFICIT = average rainfall significantly below crop water need")
    print("low     = average rainfall slightly below crop water need")
    print("—       = crop not actively growing this month")
    print("(no marker) = average rainfall meets or exceeds crop need\n")


def _month_name(month_num):
    """Convert month number to name."""
    names = ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]
    return names[month_num - 1]


# -------------------------------------------------------
# MAIN - runs when we execute this file directly
# -------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CROP SCIENCE DATABASE")
    print("Vojvodina, Serbia — No Irrigation")
    print("=" * 60)

    # Full crop summary
    list_crops()

    # Key threshold comparison
    print("\n=== KEY THRESHOLD COMPARISON ===\n")
    print(f"{'Crop':<12} {'Frost Kill':>12} {'Heat Stress':>12} "
          f"{'GDD Base':>10} {'Water(mm)':>11}")
    print("-" * 60)

    for key, crop in CROPS.items():
        frost_kill  = crop["frost"]["foliage_kill_c"]
        heat_stress = get_heat_stress_temp(key)
        gdd_base    = crop["gdd"]["base_temp_c"]
        water       = crop["water"]["total_season_mm"]
        print(f"{crop['name']:<12} "
              f"{frost_kill:>10.1f}°C "
              f"{heat_stress:>10.1f}°C "
              f"{gdd_base:>9.0f}°C "
              f"{water:>10}mm")

    # Rainfall requirements table
    show_rainfall_requirements()

    # Planting order
    print("=== PLANTING ORDER (earliest to latest) ===\n")
    print("1. Onion     → Feb/March    frost tolerant, use early rainfall")
    print("2. Potato    → March/April  some frost tolerance, avoid summer")
    print("3. Tomato    → Late Apr/May frost sensitive, needs Jun-Aug rain")
    print("4. Cucumber  → Late Apr/May most frost sensitive, heaviest water need")

    # No-irrigation risk summary
    print("\n=== NO-IRRIGATION RISK SUMMARY ===\n")
    print("Vojvodina avg July rainfall: 57mm")
    print("Vojvodina avg August rainfall: 51mm\n")

    for key, crop in CROPS.items():
        heat_temp = get_heat_stress_temp(key)
        july_need = get_monthly_water_need(key, 7)
        july_deficit = july_need - 57
        aug_need  = get_monthly_water_need(key, 8)
        aug_deficit  = aug_need - 51

        print(f"{crop['name']}")
        print(f"  July water need:   {july_need}mm  "
              f"(avg deficit: {july_deficit:+}mm)")
        print(f"  August water need: {aug_need}mm  "
              f"(avg deficit: {aug_deficit:+}mm)")
        print(f"  Heat stress above: {heat_temp}°C")
        print()

    print("=" * 60)
    print("Crop database loaded successfully.")
    print("=" * 60)
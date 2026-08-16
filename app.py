# app.py
#
# Crop Planting Advisor — Vojvodina, Serbia
#
# Produces a plain-text planting recommendation report
# combining statistical analysis and machine learning
# predictions from 26 years of NASA climate data.
#
# Run this file to get your planting recommendations:
#   python app.py
#
# Or ask about a specific crop:
#   python app.py tomato

import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")

from config import CONFIG
from crop_model import CROPS, get_crop, get_heat_stress_temp
from crop_model import get_frost_kill_temp

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

STAT_FILE = CONFIG["risk_scores_file"]
ML_FILE   = CONFIG["ml_pred_file"]

LOCATION  = CONFIG["location_name"]
DATA_FROM = CONFIG["analysis_start"]
DATA_TO   = CONFIG["analysis_end"]
NASA_SRC  = "NASA POWER (MERRA2)"

AVG_ANNUAL_RAIN_MM = 637
WARMING_PER_DECADE = 0.62
TOTAL_WARMING      = 2.73
RECENT_HEAT_DAYS   = 20.5
EARLY_HEAT_DAYS    = 7.9
AVG_LAST_FROST     = "April 5"
SAFE_FROST_DATE    = "April 24"

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def separator(char="=", width=62):
    return char * width


def risk_label(score):
    """Convert 0-1 risk score to readable label."""
    if score < 0.15:   return "Very Low"
    elif score < 0.25: return "Low"
    elif score < 0.40: return "Moderate"
    elif score < 0.55: return "High"
    else:              return "Very High"


def risk_bar(score, width=20):
    """Simple text bar showing risk level."""
    filled = int(score * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def combine_stat_ml(stat_prob, ml_prob, ml_weight=0.55):
    """
    Blend statistical and ML probabilities.
    When ML is not available, use statistical only.
    """
    if ml_prob is None or (isinstance(ml_prob, float)
                           and np.isnan(ml_prob)):
        return stat_prob
    stat_weight = 1 - ml_weight
    return (stat_prob * stat_weight) + (ml_prob * ml_weight)


def load_data():
    """Load statistical and ML prediction files."""
    if not os.path.exists(STAT_FILE):
        print("ERROR: Statistical risk scores not found.")
        print("Please run: python src/risk_model.py")
        sys.exit(1)

    if not os.path.exists(ML_FILE):
        print("WARNING: ML predictions not found.")
        print("Using statistical model only.\n")
        ml_df = None
    else:
        ml_df = pd.read_csv(ML_FILE)

    stat_df = pd.read_csv(STAT_FILE)
    return stat_df, ml_df


def get_combined_predictions(crop_key, stat_df, ml_df):
    """
    For a given crop, combine statistical and ML predictions
    into a single table of planting date risk estimates.
    """
    stat_crop = stat_df[stat_df["crop_key"] == crop_key].copy()

    if ml_df is not None:
        ml_crop = ml_df[ml_df["crop_key"] == crop_key].copy()
    else:
        ml_crop = None

    rows = []

    for _, stat_row in stat_crop.iterrows():
        month      = stat_row["month"]
        day        = stat_row["day"]
        date_label = stat_row["date_label"]

        # Get ML predictions for this date if available
        ml_frost = np.nan
        ml_heat  = np.nan
        ml_rain  = np.nan

        if ml_crop is not None and not ml_crop.empty:
            # Match on date_label which is always present
            # Fall back to month/day if date_label not available
            if "date_label" in ml_crop.columns:
                ml_match = ml_crop[
                    ml_crop["date_label"] == date_label
                ]
            elif ("month" in ml_crop.columns and
                  "day"   in ml_crop.columns):
                ml_match = ml_crop[
                    (ml_crop["month"] == month) &
                    (ml_crop["day"]   == day)
                ]
            else:
                ml_match = pd.DataFrame()

            if not ml_match.empty:
                ml_frost = ml_match["frost_kill_prob"].iloc[0] \
                    if "frost_kill_prob"  in ml_match.columns \
                    else np.nan
                ml_heat  = ml_match["heat_stress_prob"].iloc[0] \
                    if "heat_stress_prob" in ml_match.columns \
                    else np.nan
                ml_rain  = ml_match["rain_deficit_prob"].iloc[0] \
                    if "rain_deficit_prob" in ml_match.columns \
                    else np.nan

        # Combine statistical and ML
        frost_combined = combine_stat_ml(
            stat_row["frost_kill_prob"], ml_frost
        )
        heat_combined  = combine_stat_ml(
            stat_row["heat_severe_prob"], ml_heat
        )
        rain_combined  = combine_stat_ml(
            stat_row["rain_deficit_prob"], ml_rain
        )

        # Recalculate combined score with blended inputs
        crop = get_crop(crop_key)
        if crop["season_type"] == "cool":
            weights = {"frost": 0.35, "heat": 0.30, "rain": 0.35}
        else:
            weights = {"frost": 0.30, "heat": 0.28, "rain": 0.42}

        combined = (
            frost_combined * weights["frost"] +
            heat_combined  * weights["heat"]  +
            rain_combined  * weights["rain"]
        )

        # Heat + drought penalty
        combined += frost_combined * rain_combined * 0.15

        # Frost veto for warm season crops
        if (crop["season_type"] == "warm" and
                frost_combined > 0.10):
            combined += (frost_combined - 0.10) * 0.50

        combined = min(1.0, max(0.0, combined))

        rows.append({
            "date_label"  : date_label,
            "month"       : month,
            "day"         : day,
            "frost_prob"  : frost_combined,
            "heat_prob"   : heat_combined,
            "rain_prob"   : rain_combined,
            "combined"    : combined,
            "suitability" : risk_label(combined),
        })

    return pd.DataFrame(rows).sort_values(
        "combined"
    ).reset_index(drop=True)


# -------------------------------------------------------
# REPORT SECTIONS
# -------------------------------------------------------

def print_header():
    print()
    print(separator())
    print(f"  CROP PLANTING ADVISOR")
    print(f"  {LOCATION}")
    print(f"  No Irrigation — Rain-fed Agriculture")
    print(separator())
    print(f"  Data source: {NASA_SRC}")
    print(f"  Analysis period: {DATA_FROM}–{DATA_TO} "
          f"(26 years of daily climate data)")
    print(f"  Methods: Statistical analysis + "
          f"Random Forest ML")
    print(separator())


def print_climate_context():
    print()
    print(separator("-"))
    print("  CLIMATE CONTEXT FOR VOJVODINA")
    print(separator("-"))
    print()
    print(f"  Average annual rainfall:    {AVG_ANNUAL_RAIN_MM} mm")
    print(f"  Average last spring frost:  {AVG_LAST_FROST}")
    print(f"  90th percentile last frost: {SAFE_FROST_DATE}")
    print(f"  Frost-free growing season:  ~210 days")
    print()
    print(f"  WARMING TREND (1981–2025):")
    print(f"    Total warming since 1981: +{TOTAL_WARMING}°C")
    print(f"    Rate:                     "
          f"+{WARMING_PER_DECADE}°C per decade")
    print(f"    Heat days/year 1981-1995: "
          f"{EARLY_HEAT_DAYS} days above 35°C")
    print(f"    Heat days/year 2011-2025: "
          f"{RECENT_HEAT_DAYS} days above 35°C")
    print(f"    Frost days trend:         -8.8 days per decade")
    print()
    print(f"  RAINFALL CHALLENGE:")
    print(f"    June avg (best month):    75 mm")
    print(f"    July avg:                 57 mm  ← drought risk")
    print(f"    August avg:               51 mm  ← drought risk")
    print(f"    Without irrigation, July-August rainfall is")
    print(f"    critically below crop water needs for all crops.")
    print()


def print_crop_recommendation(crop_key, stat_df, ml_df):
    """Print a complete recommendation for one crop."""
    crop      = get_crop(crop_key)
    crop_name = crop["name"]
    season    = crop["season_type"]

    print()
    print(separator("="))
    print(f"  {crop_name.upper()}")
    print(f"  {crop['latin']}")
    print(f"  Season type: {season.capitalize()}-season crop")
    print(separator("="))
    print()

    frost_kill  = get_frost_kill_temp(crop_key)
    heat_stress = get_heat_stress_temp(crop_key)

    print(f"  Frost kills plant at:  {frost_kill}°C")
    print(f"  Heat stress begins at: {heat_stress}°C")
    print(f"  Water need (season):   "
          f"{crop['water']['total_season_mm']} mm")
    print(f"  Drought sensitivity:   "
          f"{crop['water']['drought_sensitivity']}")
    print(f"  No-irrigation outlook: "
          f"{crop['water']['no_irrigation_risk'][:55]}...")
    print()

    # Get predictions
    predictions = get_combined_predictions(
        crop_key, stat_df, ml_df
    )

    if predictions.empty:
        print("  No prediction data available.")
        return

    # ---------------------------------------------------
    # PLANTING DATE RISK TABLE
    # ---------------------------------------------------

    print(f"  PLANTING DATE RISK ANALYSIS")
    print(f"  Based on {DATA_FROM}-{DATA_TO} historical data")
    print(f"  Combined statistical model + ML predictions")
    print()
    print(f"  {'Date':<12} {'Frost':>7} {'Heat':>7} "
          f"{'Rain':>7} {'Overall':>9}  Risk Level")
    print(f"  {'':12} {'Risk':>7} {'Risk':>7} "
          f"{'Deficit':>7} {'Score':>9}")
    print(f"  {'-'*58}")

    for idx, row in predictions.iterrows():
        frost_pct = row["frost_prob"] * 100
        heat_pct  = row["heat_prob"]  * 100
        rain_pct  = row["rain_prob"]  * 100
        combo_pct = row["combined"]   * 100
        suit      = row["suitability"]
        marker    = " ◄ BEST" if idx == 0 else ""

        print(f"  {row['date_label']:<12} "
              f"{frost_pct:>6.0f}% "
              f"{heat_pct:>6.0f}% "
              f"{rain_pct:>6.0f}% "
              f"{combo_pct:>8.0f}%  "
              f"{suit}{marker}")

    print()

    # ---------------------------------------------------
    # TOP 3 RECOMMENDED DATES
    # ---------------------------------------------------

    best_3 = predictions.head(3)

    print(f"  RECOMMENDED PLANTING WINDOW")
    print(separator("-"))
    print()

    labels = ["Best", "Good", "Acceptable"]
    for rank, (idx, row) in enumerate(best_3.iterrows()):
        label    = labels[rank]
        combo    = row["combined"] * 100
        frost    = row["frost_prob"]  * 100
        heat     = row["heat_prob"]   * 100
        rain     = row["rain_prob"]   * 100
        bar      = risk_bar(row["combined"])

        print(f"  {rank+1}. {label}: {row['date_label']}")
        print(f"     Risk score:    {combo:.0f}%  {bar}")
        print(f"     Frost risk:    {frost:.0f}%")
        print(f"     Heat risk:     {heat:.0f}%")
        print(f"     Rain deficit:  {rain:.0f}%")
        print()

    # ---------------------------------------------------
    # PLAIN LANGUAGE ADVICE
    # ---------------------------------------------------

    print(f"  ADVICE FOR YOUR SITUATION")
    print(separator("-"))
    print()

    best = predictions.iloc[0]
    advice = get_plain_language_advice(
        crop_key, best, predictions
    )
    for line in advice:
        print(f"  {line}")
    print()


def get_plain_language_advice(crop_key, best_row, predictions):
    """Generate plain language advice for each crop."""
    crop      = get_crop(crop_key)
    best_date = best_row["date_label"]
    best_risk = best_row["combined"] * 100
    frost     = best_row["frost_prob"]  * 100
    heat      = best_row["heat_prob"]   * 100
    rain      = best_row["rain_prob"]   * 100

    lines = []

    if crop_key == "potato":
        lines += [
            f"Best planting window: late February to March 10.",
            f"",
            f"The key strategy for Vojvodina is to plant EARLY",
            f"using early varieties (70-90 day types). Early",
            f"planting allows harvest in late May to June —",
            f"before the July-August heat and drought arrives.",
            f"",
            f"Frost risk in early March is real ({frost:.0f}%) but",
            f"potatoes recover well from frost because the tuber",
            f"survives underground and sends up new shoots.",
            f"Complete crop loss from frost is rare.",
            f"",
            f"Avoid planting after April 1. By then you are",
            f"exposing the crop to peak summer heat and",
            f"summer drought (60-100% rain deficit).",
            f"",
            f"Recommended variety type: early (70-90 day).",
            f"Target harvest: late May to late June.",
        ]

    elif crop_key == "tomato":
        lines += [
            f"Best planting window: April 10 to April 20.",
            f"",
            f"IMPORTANT: There is no low-risk planting date for",
            f"tomatoes without irrigation in Vojvodina.",
            f"The model found High or Very Poor scores everywhere.",
            f"This is an honest finding, not a model error.",
            f"",
            f"The fundamental problem:",
            f"  Plant early (before April 10) → high frost risk",
            f"  Plant late (after May 10)  → peak drought exposure",
            f"",
            f"April 10-20 is the best available compromise:",
            f"  Frost risk drops significantly ({frost:.0f}%)",
            f"  Fruit develops partly in June (better rainfall)",
            f"  Heat stress is unavoidable in current climate",
            f"",
            f"What to do without irrigation:",
            f"  Use drought-tolerant varieties.",
            f"  Mulch heavily to retain soil moisture.",
            f"  Plant in a sheltered spot with morning sun.",
            f"  Expect good yields in wet summers (e.g. 2023)",
            f"  and reduced yields in dry summers (e.g. 2025).",
            f"",
            f"Rain deficit probability is near 100% every year.",
            f"Yield will vary based on June-August rainfall.",
        ]

    elif crop_key == "onion":
        lines += [
            f"Best planting window: March 1 to March 20.",
            f"",
            f"Onions are your most frost-tolerant crop.",
            f"Established plants survive down to -8°C.",
            f"This allows early planting that uses spring",
            f"rainfall before the summer drought begins.",
            f"",
            f"The key strategy is to get bulbs developing",
            f"in May-June when rainfall averages 70-75mm",
            f"rather than July when it drops to 57mm.",
            f"",
            f"Important: Use LONG-DAY varieties only.",
            f"At latitude 45°N, short-day varieties will",
            f"not form proper bulbs.",
            f"",
            f"Frost risk at March 1 is only {frost:.0f}% for killing",
            f"frost (the -8°C threshold). Onions will tolerate",
            f"the frost events that would kill tomatoes.",
            f"",
            f"Rain deficit risk is {rain:.0f}% — this is the main",
            f"challenge. Early planting is the best available",
            f"strategy to shift water demand toward wetter months.",
        ]

    elif crop_key == "cucumber":
        lines += [
            f"Best planting window: April 20 to May 1.",
            f"",
            f"The model identifies late April as the optimal",
            f"window for cucumbers — the clearest finding",
            f"in the entire analysis.",
            f"",
            f"Why late April specifically:",
            f"  Frost risk drops to near zero ({frost:.0f}%)",
            f"  The 70-day season ends in late June",
            f"  Fruit develops in June (75mm avg rainfall)",
            f"  Avoids the worst of July-August drought",
            f"",
            f"Cucumber heat stress threshold is 40°C which",
            f"is rarely exceeded — heat is less of a concern",
            f"than for tomatoes in your climate.",
            f"",
            f"Rain deficit risk is {rain:.0f}% even at this date.",
            f"This cannot be eliminated without irrigation.",
            f"",
            f"Choose fast-maturing varieties (50-60 days).",
            f"Mulch to retain moisture.",
            f"A second sowing in early July is possible",
            f"for autumn harvest if summer has been wet.",
        ]

    elif crop_key == "pepper":
        lines += [
            f"Best planting window: April 20 to May 1.",
            f"",
            f"Peppers are a traditional Vojvodina crop and",
            f"more manageable than tomatoes in current climate.",
            f"",
            f"Key advantage over tomatoes:",
            f"  Heat stress threshold is 38°C (vs 35°C tomato)",
            f"  Peppers recover better after heat events",
            f"  Heat risk at best date is only {heat:.0f}%",
            f"",
            f"Start indoors: January 15 to February 15.",
            f"Peppers need 8-10 weeks indoors before transplant.",
            f"This is earlier than tomatoes — plan accordingly.",
            f"",
            f"Rain deficit risk is {rain:.0f}% at best date.",
            f"Without irrigation yields will vary year to year.",
            f"Compared to tomatoes, peppers are a better",
            f"choice for rain-fed growing in Vojvodina.",
        ]

    elif crop_key == "watermelon":
        lines += [
            f"Best planting window: May 1 to May 10.",
            f"",
            f"Watermelon is well suited to rain-fed growing",
            f"in Vojvodina. Its deep root system accesses",
            f"subsoil moisture that shallow-rooted crops cannot.",
            f"",
            f"Key advantages without irrigation:",
            f"  Deep tap root = more drought tolerant",
            f"  Heat tolerance up to 42°C",
            f"  Heat risk at best date: {heat:.0f}%",
            f"  Lower water need than tomatoes or cucumbers",
            f"",
            f"Needs warm soil above 18°C to germinate well.",
            f"In Vojvodina this typically means early May.",
            f"Can be transplanted or direct seeded from mid-May.",
            f"",
            f"Rain deficit risk is {rain:.0f}% — still present",
            f"but more manageable than for tomatoes or corn.",
            f"In dry years watermelons will survive where",
            f"tomatoes and cucumbers would fail.",
        ]

    elif crop_key == "sunflower":
        lines += [
            f"Best planting window: April 10 to May 10.",
            f"",
            f"Sunflower is the best warm-season crop for",
            f"rain-fed growing in Vojvodina. It is specifically",
            f"adapted to exactly this climate.",
            f"",
            f"Why sunflower handles drought best:",
            f"  Very deep tap root — accesses subsoil water",
            f"  Heat tolerant up to 42°C",
            f"  Lower total water need (400mm vs 600mm corn)",
            f"  Frost risk at best date: {frost:.0f}%",
            f"  Heat risk at best date:  {heat:.0f}%",
            f"",
            f"Traditional Vojvodina crop — varieties bred",
            f"specifically for this region are available",
            f"from local seed suppliers.",
            f"",
            f"If expanding beyond garden to small-scale field",
            f"production without irrigation, sunflower is",
            f"the most reliable option.",
        ]

    elif crop_key == "corn":
        lines += [
            f"Best planting window: April 20 to May 1.",
            f"",
            f"IMPORTANT: Corn without irrigation in Vojvodina",
            f"is increasingly risky given current warming trends.",
            f"Combined risk: {best_risk:.0f}% — highest of all crops.",
            f"",
            f"The critical problem — double threat:",
            f"  1. Pollen non-viable above 38°C during silking",
            f"  2. Peak water need (160mm) in July when",
            f"     Vojvodina averages only 57mm rainfall",
            f"",
            f"Heat risk at best date:  {heat:.0f}%",
            f"Rain deficit at best date: {rain:.0f}%",
            f"",
            f"Strategy if growing corn without irrigation:",
            f"  Use early hybrids (FAO 200-300) that silk",
            f"  in June rather than July — avoids peak heat",
            f"  Plant April 20 or earlier to shift silking",
            f"  into the cooler wetter June period",
            f"  Accept yield variability of 30-60% between",
            f"  wet and dry years",
            f"",
            f"Without irrigation, sunflower is a more reliable",
            f"alternative for field-scale production.",
        ]

    else:
        lines += [
            f"Best planting window: {best_date}.",
            f"",
            f"Combined risk at best date: {best_risk:.0f}%",
            f"  Frost risk:   {frost:.0f}%",
            f"  Heat risk:    {heat:.0f}%",
            f"  Rain deficit: {rain:.0f}%",
            f"",
            f"Without irrigation, rainfall during critical",
            f"growth months is the main limiting factor.",
            f"Early planting that uses spring rainfall",
            f"before the summer drought is generally",
            f"the most effective strategy.",
        ]

    return lines


def print_combined_summary(stat_df, ml_df):
    """Print a single-page summary of all crops."""
    print()
    print(separator("="))
    print("  QUICK REFERENCE SUMMARY — ALL CROPS")
    print(f"  {LOCATION} — No Irrigation")
    print(separator("="))
    print()

    crop_summaries = []

    for crop_key in CROPS.keys():
        predictions = get_combined_predictions(
            crop_key, stat_df, ml_df
        )
        if predictions.empty:
            continue

        best      = predictions.iloc[0]
        crop_name = CROPS[crop_key]["name"]

        crop_summaries.append({
            "crop"       : crop_name,
            "crop_key"   : crop_key,
            "best_date"  : best["date_label"],
            "score"      : best["combined"] * 100,
            "suitability": best["suitability"],
            "frost"      : best["frost_prob"]  * 100,
            "heat"       : best["heat_prob"]   * 100,
            "rain"       : best["rain_prob"]   * 100,
        })

    # Sort by combined score — best first
    crop_summaries.sort(key=lambda x: x["score"])

    print(f"  {'Crop':<14} {'Best Date':<12} "
          f"{'Frost':>7} {'Heat':>7} {'Rain':>7} "
          f"{'Score':>7}  Outlook")
    print(f"  {'-'*70}")

    for s in crop_summaries:
        print(f"  {s['crop']:<14} {s['best_date']:<12} "
              f"{s['frost']:>6.0f}% "
              f"{s['heat']:>6.0f}% "
              f"{s['rain']:>6.0f}% "
              f"{s['score']:>6.0f}%  {s['suitability']}")

    print()
    print(separator("-"))
    print()
    print("  PLANTING CALENDAR")
    print()
    print("  February 15 onwards:")
    print("    Onion sets (frost tolerant, use spring rainfall)")
    print()
    print("  March 1 — March 10:")
    print("    Potato — early varieties "
          "(key: harvest before July)")
    print("    Onion  — transplants or direct seed")
    print()
    print("  March 21 — April 10:")
    print("    Sunflower (drought tolerant, best field crop)")
    print()
    print("  April 1 — April 4:")
    print("    Corn — early hybrids FAO 200-300 only")
    print()
    print("  April 20 — May 1:")
    print("    Cucumber    (April 20 optimal — uses June rain)")
    print("    Pepper      (frost risk cleared, "
          "start indoors Jan)")
    print("    Tomato      (no safe date — accept yield risk)")
    print("    Watermelon  (May 1 optimal — needs warm soil)")
    print()
    print(separator("-"))
    print()
    print("  RISK RANKING (most to least manageable)")
    print("  without irrigation in current Vojvodina climate:")
    print()
    for i, s in enumerate(crop_summaries, 1):
        bar = risk_bar(s["score"] / 100, width=10)
        print(f"  {i}. {s['crop']:<14} {bar}  "
              f"{s['score']:.0f}%  {s['suitability']}")
    print()


def print_soil_temperature_note():
    """
    Print soil temperature readiness dates if available.
    """
    rfile = "data/processed/soil_readiness_dates.csv"
    if not os.path.exists(rfile):
        return

    rdf = pd.read_csv(rfile)

    print()
    print(separator("="))
    print("  SOIL TEMPERATURE READINESS")
    print("  When does soil reach germination temperature?")
    print("  (NASA POWER TSOIL data, 0-10cm depth)")
    print(separator("="))
    print()
    print(f"  {'Crop':<14} {'Min Soil':>9} "
          f"{'Avg Ready':>11} {'Safe (90%)':>12}")
    print(f"  {'-'*50}")

    def doy_to_str(doy):
        return (pd.Timestamp("2024-01-01") +
                pd.Timedelta(days=int(doy) - 1)).strftime("%b %d")

    for crop_key in CROPS.keys():
        crop_data = rdf[rdf["crop_key"] == crop_key]
        if crop_data.empty:
            continue

        crop_name = CROPS[crop_key]["name"]
        min_soil  = crop_data["min_soil_c"].iloc[0]
        avg_doy   = crop_data["day_of_year"].mean()
        p90_doy   = crop_data["day_of_year"].quantile(0.90)

        print(f"  {crop_name:<14} "
              f"{min_soil:>8.0f}°C "
              f"{doy_to_str(avg_doy):>11} "
              f"{doy_to_str(p90_doy):>12}")

    print()
    print("  Avg Ready = average date soil reaches threshold")
    print("  Safe (90%) = soil ready in 90% of years by this date")
    print()
    print("  Note: Air-temperature-based recommendations may")
    print("  suggest planting before soil is warm enough.")
    print("  When in doubt check actual soil temperature at")
    print("  seeding depth before planting.")
    print()


def print_disclaimer():
    print(separator("="))
    print("  IMPORTANT — HOW TO USE THESE RECOMMENDATIONS")
    print(separator("="))
    print()
    print("  These recommendations are based on 26 years of")
    print("  historical NASA climate data for your location,")
    print("  combined with machine learning models trained on")
    print("  that data.")
    print()
    print("  What these numbers mean:")
    print("    Frost risk   = probability of killing frost in 21")
    print("                   days after planting")
    print("    Heat risk    = probability of sustained damaging")
    print("                   heat during the growing season")
    print("    Rain deficit = probability of insufficient rainfall")
    print("                   in critical growth months")
    print()
    print("  What these numbers do NOT mean:")
    print("    They are NOT a weather forecast.")
    print("    They cannot predict what this specific year brings.")
    print("    A 10% frost risk means 1 in 10 years had frost —")
    print("    that year might be this year.")
    print()
    print("  Warming trend note:")
    print("    Vojvodina has warmed +2.73°C since 1981.")
    print("    Spring frost risk is decreasing.")
    print("    Summer heat and drought risk is increasing.")
    print("    These trends are incorporated in ML predictions.")
    print()
    print("  Always combine with local knowledge.")
    print("  Watch actual soil temperature before planting.")
    print("  Monitor the 10-day forecast around planting date.")
    print()
    print(separator("="))
    print()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    # Check if user asked about a specific crop
    requested_crop = None
    if len(sys.argv) > 1:
        requested_crop = sys.argv[1].lower().strip()
        if requested_crop not in CROPS:
            print(f"\nCrop '{requested_crop}' not found.")
            print(f"Available crops: {list(CROPS.keys())}")
            sys.exit(1)

    # Load prediction data
    stat_df, ml_df = load_data()

    # Print report
    print_header()
    print_climate_context()

    if requested_crop:
        print_crop_recommendation(
            requested_crop, stat_df, ml_df
        )
    else:
        for crop_key in CROPS.keys():
            print_crop_recommendation(
                crop_key, stat_df, ml_df
            )
        print_combined_summary(stat_df, ml_df)

    print_soil_temperature_note()
    print_disclaimer()


if __name__ == "__main__":
    main()
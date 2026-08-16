# src/climate_analysis.py
#
# Phase 6 - Statistical Climate Analysis for Vojvodina, Serbia
#
# We analyze 40+ years of NASA POWER data to understand
# the historical climate of our region.
#
# This is the foundation of everything else in this project.
# We must understand the climate statistically BEFORE we
# make any crop recommendations or use machine learning.
#
# What this script calculates:
# 1. Monthly temperature statistics
# 2. Seasonal patterns
# 3. Frost day analysis
# 4. Heat day analysis
# 5. Rainfall patterns
# 6. Growing season statistics

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE = "data/processed/vojvodina_clean.csv"

# -------------------------------------------------------
# AGRICULTURAL THRESHOLDS
# -------------------------------------------------------
# These are scientifically recognized temperature thresholds
# used in agricultural climate analysis.
# Sources: FAO Agricultural Guidelines, USDA Climate Data

FROST_THRESHOLD    =  0.0   # °C - temperature at or below which frost can occur
HARD_FROST         = -4.0   # °C - damaging to most crops
HEAT_STRESS        = 35.0   # °C - heat stress threshold for most field crops
EXTREME_HEAT       = 38.0   # °C - extreme heat, damaging to most crops

# -------------------------------------------------------
# STEP 1 - LOAD CLEAN DATA
# -------------------------------------------------------

print("Loading clean data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

# How many years of data do we have?
years = df["year"].unique()
n_years = len(years)
print(f"Loaded {len(df):,} days across {n_years} years")
print(f"Years: {years.min()} to {years.max()}")
print("-" * 60)

# -------------------------------------------------------
# STEP 2 - MONTHLY TEMPERATURE STATISTICS
# -------------------------------------------------------
# For each month (1=January to 12=December) calculate
# the average, minimum, and maximum temperature across
# all years in our dataset.

print("\n=== MONTHLY TEMPERATURE STATISTICS (1981-2024) ===\n")

# Group all rows by month number, then calculate statistics
monthly_temp = df.groupby("month").agg(
    avg_temp        = ("temp_avg", "mean"),   # average of daily averages
    avg_max         = ("temp_max", "mean"),   # average of daily maximums
    avg_min         = ("temp_min", "mean"),   # average of daily minimums
    record_high     = ("temp_max", "max"),    # highest temperature ever recorded
    record_low      = ("temp_min", "min"),    # lowest temperature ever recorded
    std_temp        = ("temp_avg", "std"),    # standard deviation
).round(2)

# Give months proper names for readability
month_names = {
    1:"January", 2:"February", 3:"March",    4:"April",
    5:"May",     6:"June",     7:"July",     8:"August",
    9:"September",10:"October",11:"November",12:"December"
}
monthly_temp.index = monthly_temp.index.map(month_names)

print(monthly_temp.to_string())

# -------------------------------------------------------
# STEP 3 - EXPLAIN STANDARD DEVIATION
# -------------------------------------------------------
# Standard deviation tells us how much temperatures vary
# around the average. A high standard deviation means
# temperatures are unpredictable in that month.
# A low standard deviation means temperatures are consistent.

print("\n--- TEMPERATURE VARIABILITY (Standard Deviation) ---")
print("\nThe standard deviation tells us how reliable the average is.")
print("A HIGH value means temperatures vary a lot from year to year.")
print("A LOW value means temperatures are fairly consistent.\n")

for month, row in monthly_temp.iterrows():
    print(f"{month:<12} avg={row['avg_temp']:>6.1f}°C   "
          f"variability=±{row['std_temp']:.1f}°C")

# -------------------------------------------------------
# STEP 4 - FROST DAY ANALYSIS
# -------------------------------------------------------
# A frost day is any day where the minimum temperature
# drops to 0°C or below.
# This is critical for crop planting decisions.

print("\n\n=== FROST DAY ANALYSIS ===\n")

# Find all frost days
frost_days = df[df["temp_min"] <= FROST_THRESHOLD]
hard_frost_days = df[df["temp_min"] <= HARD_FROST]

print(f"Total frost days in dataset: {len(frost_days):,}")
print(f"Total hard frost days (below {HARD_FROST}°C): "
      f"{len(hard_frost_days):,}")
print(f"Average frost days per year: "
      f"{len(frost_days)/n_years:.1f}")
print(f"Average hard frost days per year: "
      f"{len(hard_frost_days)/n_years:.1f}")

# Monthly frost frequency
print("\n--- FROST DAYS BY MONTH ---")
print("(average number of frost days per month across all years)\n")

monthly_frost = df[df["temp_min"] <= FROST_THRESHOLD]\
    .groupby("month").size() / n_years

monthly_frost.index = monthly_frost.index.map(month_names)

for month, count in monthly_frost.items():
    # Create a simple visual bar
    bar = "█" * int(count)
    print(f"{month:<12} {count:>5.1f} days   {bar}")

# -------------------------------------------------------
# STEP 5 - LAST SPRING FROST AND FIRST AUTUMN FROST
# -------------------------------------------------------
# For each year, find:
# - The LAST frost day of spring (most important for planting)
# - The FIRST frost day of autumn (important for harvest)
# These define the "frost-free growing season"

print("\n\n=== FROST-FREE GROWING SEASON ===\n")

last_spring_frosts  = []
first_autumn_frosts = []

for year in years:
    # Get all frost days in this specific year
    year_data = df[
        (df["year"] == year) &
        (df["temp_min"] <= FROST_THRESHOLD)
    ]

    if len(year_data) == 0:
        continue

    # Spring = days before July (day of year < 182)
    spring_frosts = year_data[year_data["date"].dt.dayofyear < 182]
    # Autumn = days from July onwards
    autumn_frosts = year_data[year_data["date"].dt.dayofyear >= 182]

    if len(spring_frosts) > 0:
        # The LAST frost in spring is the most dangerous for planting
        last_spring = spring_frosts["date"].max()
        last_spring_frosts.append({
            "year"       : year,
            "date"       : last_spring,
            "day_of_year": last_spring.dayofyear,
            "month_day"  : last_spring.strftime("%b %d")
        })

    if len(autumn_frosts) > 0:
        # The FIRST frost in autumn ends the growing season
        first_autumn = autumn_frosts["date"].min()
        first_autumn_frosts.append({
            "year"       : year,
            "date"       : first_autumn,
            "day_of_year": first_autumn.dayofyear,
            "month_day"  : first_autumn.strftime("%b %d")
        })

# Convert to DataFrames
spring_df = pd.DataFrame(last_spring_frosts)
autumn_df = pd.DataFrame(first_autumn_frosts)

# Last spring frost statistics
print("--- LAST SPRING FROST ---")
print(f"Average last spring frost:  day {spring_df['day_of_year'].mean():.0f} "
      f"of year = around "
      f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(spring_df['day_of_year'].mean())-1):%B %d}")
print(f"Earliest last spring frost: {spring_df.loc[spring_df['day_of_year'].idxmin(), 'month_day']} "
      f"({spring_df.loc[spring_df['day_of_year'].idxmin(), 'year']:.0f})")
print(f"Latest last spring frost:   {spring_df.loc[spring_df['day_of_year'].idxmax(), 'month_day']} "
      f"({spring_df.loc[spring_df['day_of_year'].idxmax(), 'year']:.0f})")

# 10th, 50th, 90th percentile
p10 = spring_df["day_of_year"].quantile(0.10)
p50 = spring_df["day_of_year"].quantile(0.50)
p90 = spring_df["day_of_year"].quantile(0.90)

print(f"\n10th percentile: day {p10:.0f} — "
      f"In 90% of years, there is still a frost after this date")
print(f"50th percentile: day {p50:.0f} — "
      f"In 50% of years, the last frost is around this date")
print(f"90th percentile: day {p90:.0f} — "
      f"In only 10% of years does frost occur after this date")

# First autumn frost statistics
print("\n--- FIRST AUTUMN FROST ---")
print(f"Average first autumn frost: day {autumn_df['day_of_year'].mean():.0f} "
      f"of year = around "
      f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(autumn_df['day_of_year'].mean())-1):%B %d}")

# Growing season length
print("\n--- FROST-FREE GROWING SEASON ---")
season_lengths = autumn_df["day_of_year"].values - spring_df["day_of_year"].values[:len(autumn_df)]
print(f"Average growing season length: {season_lengths.mean():.0f} days")
print(f"Shortest growing season: {season_lengths.min():.0f} days")
print(f"Longest growing season:  {season_lengths.max():.0f} days")

# -------------------------------------------------------
# STEP 6 - HEAT DAY ANALYSIS
# -------------------------------------------------------

print("\n\n=== HEAT DAY ANALYSIS ===\n")

heat_days    = df[df["temp_max"] >= HEAT_STRESS]
extreme_days = df[df["temp_max"] >= EXTREME_HEAT]

print(f"Days above {HEAT_STRESS}°C per year: "
      f"{len(heat_days)/n_years:.1f}")
print(f"Days above {EXTREME_HEAT}°C per year: "
      f"{len(extreme_days)/n_years:.1f}")

print("\n--- HEAT DAYS BY MONTH ---")
print(f"(average days per month above {HEAT_STRESS}°C)\n")

monthly_heat = df[df["temp_max"] >= HEAT_STRESS]\
    .groupby("month").size() / n_years

monthly_heat.index = monthly_heat.index.map(month_names)

for month, count in monthly_heat.items():
    bar = "█" * int(count)
    print(f"{month:<12} {count:>5.1f} days   {bar}")

# -------------------------------------------------------
# STEP 7 - RAINFALL PATTERNS
# -------------------------------------------------------

print("\n\n=== RAINFALL PATTERNS ===\n")

monthly_rain = df.groupby("month").agg(
    avg_monthly_rain = ("precipitation", "sum"),
    rainy_days       = ("precipitation", lambda x: (x > 1.0).sum()),
    dry_days         = ("precipitation", lambda x: (x == 0).sum()),
).round(1)

# Divide by number of years to get per-year averages
monthly_rain["avg_monthly_rain"] = \
    (monthly_rain["avg_monthly_rain"] / n_years).round(1)
monthly_rain["avg_rainy_days"] = \
    (monthly_rain["rainy_days"] / n_years).round(1)

monthly_rain.index = monthly_rain.index.map(month_names)

print(f"{'Month':<12} {'Avg Rain(mm)':>14} {'Rainy Days':>12}")
print("-" * 42)
for month, row in monthly_rain.iterrows():
    print(f"{month:<12} {row['avg_monthly_rain']:>14.1f} "
          f"{row['avg_rainy_days']:>12.1f}")

total_annual = monthly_rain["avg_monthly_rain"].sum()
print(f"\nAverage annual rainfall: {total_annual:.1f} mm")

# -------------------------------------------------------
# STEP 8 - SAVE SUMMARY STATISTICS
# -------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

# Save monthly statistics for use in other scripts
monthly_temp.to_csv("data/processed/monthly_temperature_stats.csv")
monthly_rain.to_csv("data/processed/monthly_rainfall_stats.csv")
spring_df.to_csv("data/processed/last_spring_frosts.csv", index=False)

print("\n\nSummary statistics saved to data/processed/")
print("\n=== ANALYSIS COMPLETE ===")


# -------------------------------------------------------
# PHASE 7 - CLIMATE TREND ANALYSIS
# -------------------------------------------------------
# We investigate whether temperatures in Vojvodina have
# changed meaningfully over the 44-year period 1981-2024.
#
# Tools we use:
# - scipy.stats.linregress: fits a straight line through data
# - slope: how much temperature changes per year
# - p-value: how confident we are the trend is real
# - r-squared: how well the straight line fits the data
#
# IMPORTANT: We will not overstate our conclusions.
# A trend in 44 years of data is suggestive but cannot
# tell us exactly what will happen in the future.

from scipy import stats  # statistical functions

print("\n\n" + "=" * 60)
print("PHASE 7 - CLIMATE TREND ANALYSIS")
print("=" * 60)

# -------------------------------------------------------
# STEP 1 - ANNUAL AVERAGE TEMPERATURES
# -------------------------------------------------------
# First calculate one average temperature per year.
# This gives us 44 data points — one per year from
# 1981 to 2024.

annual_temp = df.groupby("year").agg(
    avg_temp    = ("temp_avg",  "mean"),
    avg_max     = ("temp_max",  "mean"),
    avg_min     = ("temp_min",  "mean"),
    total_rain  = ("precipitation", "sum"),
    frost_days  = ("temp_min",  lambda x: (x <= 0).sum()),
    heat_days   = ("temp_max",  lambda x: (x >= 35).sum()),
).round(3)

print("\n--- ANNUAL AVERAGES (first and last 5 years) ---\n")
print(annual_temp.head())
print("...")
print(annual_temp.tail())

# -------------------------------------------------------
# STEP 2 - TEMPERATURE TREND: FULL PERIOD 1981-2024
# -------------------------------------------------------

print("\n\n--- TEMPERATURE TREND 1981-2024 ---\n")

# linregress needs two lists of numbers
# x = years (1981, 1982, ..., 2024)
# y = average temperature for each year
x = annual_temp.index.values          # years
y = annual_temp["avg_temp"].values     # temperatures

# Fit the line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# How much warming per decade?
warming_per_decade = slope * 10

print(f"Temperature trend: {slope:.4f}°C per year")
print(f"                   {warming_per_decade:.3f}°C per decade")
print(f"R-squared:         {r_value**2:.3f}")
print(f"P-value:           {p_value:.4f}")
print(f"Standard error:    {std_err:.4f}")

# Explain what these numbers mean
print("\n--- WHAT THESE NUMBERS MEAN ---\n")

print(f"Slope: {slope:.4f}°C per year")
print(f"  → Each year, the average temperature has shifted")
print(f"    by approximately {slope:.4f}°C")
print(f"  → Over 10 years that is {warming_per_decade:.2f}°C")
print(f"  → Over the full 44 years: {slope*44:.2f}°C total shift")

print(f"\nR-squared: {r_value**2:.3f}")
print(f"  → {r_value**2*100:.1f}% of the year-to-year temperature")
print(f"    variation is explained by the long-term trend.")
print(f"  → The remaining {(1-r_value**2)*100:.1f}% is natural variability.")

print(f"\nP-value: {p_value:.4f}")
if p_value < 0.01:
    confidence = "very strong — the trend is almost certainly real"
elif p_value < 0.05:
    confidence = "strong — the trend is likely real"
elif p_value < 0.10:
    confidence = "moderate — the trend is suggestive but uncertain"
else:
    confidence = "weak — we cannot confidently claim a real trend"
print(f"  → Statistical confidence: {confidence}")
print(f"  → A p-value below 0.05 is the conventional threshold")
print(f"    for calling a trend statistically significant.")

# -------------------------------------------------------
# STEP 3 - COMPARE THREE PERIODS
# -------------------------------------------------------
# Instead of just fitting a line, let's compare actual
# average temperatures across three periods:
#
# Early period:  1981-1995  (first 15 years)
# Middle period: 1996-2010  (middle 15 years)
# Recent period: 2010-2024  (last 15 years)
#
# This shows whether warming has been gradual or accelerating.

print("\n\n--- THREE-PERIOD COMPARISON ---\n")
print("Comparing early, middle, and recent climate periods.\n")

periods = {
    "Early  (1981-1995)" : (1981, 1995),
    "Middle (1996-2010)" : (1996, 2010),
    "Recent (2011-2024)" : (2011, 2024),
}

period_stats = {}

for period_name, (start, end) in periods.items():
    period_data = df[
        (df["year"] >= start) &
        (df["year"] <= end)
    ]

    stats_dict = {
        "avg_temp"    : period_data["temp_avg"].mean(),
        "avg_max"     : period_data["temp_max"].mean(),
        "avg_min"     : period_data["temp_min"].mean(),
        "total_rain"  : period_data["precipitation"].sum() / len(period_data["year"].unique()),
        "frost_days"  : (period_data["temp_min"] <= 0).sum() / len(period_data["year"].unique()),
        "heat_days"   : (period_data["temp_max"] >= 35).sum() / len(period_data["year"].unique()),
    }
    period_stats[period_name] = stats_dict

    print(f"{period_name}")
    print(f"  Average temperature:    {stats_dict['avg_temp']:.2f}°C")
    print(f"  Average daily maximum:  {stats_dict['avg_max']:.2f}°C")
    print(f"  Average daily minimum:  {stats_dict['avg_min']:.2f}°C")
    print(f"  Annual rainfall:        {stats_dict['total_rain']:.1f} mm")
    print(f"  Frost days per year:    {stats_dict['frost_days']:.1f}")
    print(f"  Heat days per year:     {stats_dict['heat_days']:.1f}")
    print()

# Calculate the change from early to recent
early  = period_stats["Early  (1981-1995)"]
recent = period_stats["Recent (2011-2024)"]

print("--- CHANGE FROM EARLY TO RECENT PERIOD ---\n")
print(f"Average temperature:  "
      f"{early['avg_temp']:.2f}°C → {recent['avg_temp']:.2f}°C   "
      f"(+{recent['avg_temp']-early['avg_temp']:.2f}°C)")
print(f"Average maximum:      "
      f"{early['avg_max']:.2f}°C → {recent['avg_max']:.2f}°C   "
      f"(+{recent['avg_max']-early['avg_max']:.2f}°C)")
print(f"Average minimum:      "
      f"{early['avg_min']:.2f}°C → {recent['avg_min']:.2f}°C   "
      f"(+{recent['avg_min']-early['avg_min']:.2f}°C)")
print(f"Annual rainfall:      "
      f"{early['total_rain']:.1f}mm → {recent['total_rain']:.1f}mm   "
      f"({recent['total_rain']-early['total_rain']:+.1f}mm)")
print(f"Frost days per year:  "
      f"{early['frost_days']:.1f} → {recent['frost_days']:.1f}   "
      f"({recent['frost_days']-early['frost_days']:+.1f} days)")
print(f"Heat days per year:   "
      f"{early['heat_days']:.1f} → {recent['heat_days']:.1f}   "
      f"({recent['heat_days']-early['heat_days']:+.1f} days)")

# -------------------------------------------------------
# STEP 4 - MONTHLY TREND ANALYSIS
# -------------------------------------------------------
# The overall annual trend can hide important seasonal
# patterns. Some months may be warming faster than others.
# This matters enormously for planting decisions.

print("\n\n--- WARMING TREND BY MONTH ---\n")
print("Which months are warming fastest?\n")
print(f"{'Month':<12} {'Trend':>10} {'Per Decade':>12} {'Confidence':>15}")
print("-" * 55)

month_trends = {}

for month_num in range(1, 13):
    month_name = month_names[month_num]

    # Get annual average for this specific month
    month_data = df[df["month"] == month_num].groupby("year")["temp_avg"].mean()

    x_m = month_data.index.values
    y_m = month_data.values

    s, i, r, p, se = stats.linregress(x_m, y_m)

    # Confidence label
    if p < 0.01:
        conf = "Very strong ✓✓"
    elif p < 0.05:
        conf = "Strong ✓"
    elif p < 0.10:
        conf = "Moderate ~"
    else:
        conf = "Weak ✗"

    month_trends[month_name] = {
        "slope"   : s,
        "p_value" : p
    }

    print(f"{month_name:<12} {s:>+8.4f}°C/yr "
          f"{s*10:>+8.3f}°C/dec "
          f"{conf:>15}")

# -------------------------------------------------------
# STEP 5 - FROST TREND
# -------------------------------------------------------
# Are frost days becoming less common over time?
# This would suggest springs are arriving earlier.

print("\n\n--- FROST DAY TREND ---\n")

frost_by_year = df.groupby("year").apply(
    lambda x: (x["temp_min"] <= 0).sum()
).reset_index()
frost_by_year.columns = ["year", "frost_days"]

x_f = frost_by_year["year"].values
y_f = frost_by_year["frost_days"].values

s_f, i_f, r_f, p_f, se_f = stats.linregress(x_f, y_f)

print(f"Frost days trend: {s_f:.3f} days per year")
print(f"                  {s_f*10:.2f} days per decade")
print(f"P-value:          {p_f:.4f}")

if p_f < 0.05:
    direction = "decreasing" if s_f < 0 else "increasing"
    print(f"Conclusion: Frost days are statistically "
          f"significantly {direction}.")
else:
    print(f"Conclusion: No statistically significant trend "
          f"in frost days detected.")

# -------------------------------------------------------
# STEP 6 - HEAT DAY TREND
# -------------------------------------------------------

print("\n--- HEAT DAY TREND ---\n")

heat_by_year = df.groupby("year").apply(
    lambda x: (x["temp_max"] >= 35).sum()
).reset_index()
heat_by_year.columns = ["year", "heat_days"]

x_h = heat_by_year["year"].values
y_h = heat_by_year["heat_days"].values

s_h, i_h, r_h, p_h, se_h = stats.linregress(x_h, y_h)

print(f"Heat days trend:  {s_h:.3f} days per year")
print(f"                  {s_h*10:.2f} days per decade")
print(f"P-value:          {p_h:.4f}")

if p_h < 0.05:
    direction = "increasing" if s_h > 0 else "decreasing"
    print(f"Conclusion: Heat days are statistically "
          f"significantly {direction}.")
else:
    print(f"Conclusion: No statistically significant trend "
          f"in heat days detected.")

# -------------------------------------------------------
# STEP 7 - SAVE TREND DATA
# -------------------------------------------------------

annual_temp.to_csv("data/processed/annual_temperature_trends.csv")
frost_by_year.to_csv("data/processed/frost_days_by_year.csv",
                     index=False)
heat_by_year.to_csv("data/processed/heat_days_by_year.csv",
                    index=False)

print("\n\nTrend data saved to data/processed/")
print("\n=== TREND ANALYSIS COMPLETE ===")
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
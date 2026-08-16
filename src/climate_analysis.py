# src/climate_analysis.py
#
# Phase 6 - Statistical Climate Analysis for Vojvodina, Serbia
# Phase 7 - Climate Trend Analysis
#
# We analyze 45+ years of NASA POWER data to understand
# the historical climate of our region.
#
# IMPORTANT: Partial years (like 2026 with data only through
# July) are excluded from annual trend calculations to avoid
# misleading statistics.

import pandas as pd
import numpy as np
import os
from scipy import stats

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE = "data/processed/vojvodina_clean.csv"

# -------------------------------------------------------
# AGRICULTURAL THRESHOLDS
# -------------------------------------------------------
# Sources: FAO Agricultural Guidelines, USDA Climate Data

FROST_THRESHOLD = 0.0    # °C - frost can occur at or below this
HARD_FROST      = -4.0   # °C - damaging to most crops
HEAT_STRESS     = 35.0   # °C - heat stress for most field crops
EXTREME_HEAT    = 38.0   # °C - damaging to most crops

# Month name lookup
month_names = {
    1:"January",  2:"February", 3:"March",    4:"April",
    5:"May",      6:"June",     7:"July",     8:"August",
    9:"September",10:"October", 11:"November",12:"December"
}

# -------------------------------------------------------
# STEP 1 - LOAD CLEAN DATA
# -------------------------------------------------------

print("Loading clean data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

years   = df["year"].unique()
n_years = len(years)

print(f"Loaded {len(df):,} days across {n_years} years")
print(f"Years: {years.min()} to {years.max()}")
print("-" * 60)

# -------------------------------------------------------
# STEP 2 - IDENTIFY COMPLETE VS PARTIAL YEARS
# -------------------------------------------------------
# A partial year (like 2026 with data only through July)
# must be excluded from annual statistics.
# We keep partial years in the full dataset for monthly
# analysis but exclude them from annual trend calculations.

days_per_year   = df.groupby("year").size()
complete_years  = days_per_year[days_per_year >= 350].index
partial_years   = sorted(set(df["year"].unique()) -
                         set(complete_years.tolist()))

# df_complete only contains years with full data
df_complete = df[df["year"].isin(complete_years)]

n_complete_years = len(complete_years)

print(f"Complete years (>=350 days): "
      f"{df_complete['year'].min()} to {df_complete['year'].max()}")
if partial_years:
    print(f"Partial years excluded from annual stats: {partial_years}")
print("-" * 60)

# -------------------------------------------------------
# STEP 3 - MONTHLY TEMPERATURE STATISTICS
# -------------------------------------------------------
# Use full dataset for monthly stats (all years including partial)
# Monthly stats are not affected by partial years because
# we group by month, not by full year.

print("\n=== MONTHLY TEMPERATURE STATISTICS ===\n")

monthly_temp = df_complete.groupby("month").agg(
    avg_temp    = ("temp_avg", "mean"),
    avg_max     = ("temp_max", "mean"),
    avg_min     = ("temp_min", "mean"),
    record_high = ("temp_max", "max"),
    record_low  = ("temp_min", "min"),
    std_temp    = ("temp_avg", "std"),
).round(2)

monthly_temp.index = monthly_temp.index.map(month_names)
print(monthly_temp.to_string())

# -------------------------------------------------------
# STEP 4 - TEMPERATURE VARIABILITY
# -------------------------------------------------------

print("\n--- TEMPERATURE VARIABILITY (Standard Deviation) ---")
print("\nA HIGH value means temperatures vary a lot year to year.")
print("A LOW value means temperatures are fairly consistent.\n")

for month, row in monthly_temp.iterrows():
    print(f"{month:<12} avg={row['avg_temp']:>6.1f}°C   "
          f"variability=±{row['std_temp']:.1f}°C")

# -------------------------------------------------------
# STEP 5 - FROST DAY ANALYSIS
# -------------------------------------------------------

print("\n\n=== FROST DAY ANALYSIS ===\n")

frost_days      = df_complete[df_complete["temp_min"] <= FROST_THRESHOLD]
hard_frost_days = df_complete[df_complete["temp_min"] <= HARD_FROST]

print(f"Total frost days in dataset:              "
      f"{len(frost_days):,}")
print(f"Total hard frost days (below {HARD_FROST}°C):   "
      f"{len(hard_frost_days):,}")
print(f"Average frost days per year:              "
      f"{len(frost_days)/n_complete_years:.1f}")
print(f"Average hard frost days per year:         "
      f"{len(hard_frost_days)/n_complete_years:.1f}")

print("\n--- FROST DAYS BY MONTH ---")
print("(average number of frost days per month)\n")

monthly_frost = (df_complete[df_complete["temp_min"] <= FROST_THRESHOLD]
                 .groupby("month").size() / n_complete_years)
monthly_frost.index = monthly_frost.index.map(month_names)

for month, count in monthly_frost.items():
    bar = "█" * int(count)
    print(f"{month:<12} {count:>5.1f} days   {bar}")

# -------------------------------------------------------
# STEP 6 - LAST SPRING FROST AND FIRST AUTUMN FROST
# -------------------------------------------------------

print("\n\n=== FROST-FREE GROWING SEASON ===\n")

last_spring_frosts  = []
first_autumn_frosts = []

for year in complete_years:
    year_data = df_complete[
        (df_complete["year"] == year) &
        (df_complete["temp_min"] <= FROST_THRESHOLD)
    ]

    if len(year_data) == 0:
        continue

    spring_frosts = year_data[year_data["date"].dt.dayofyear < 182]
    autumn_frosts = year_data[year_data["date"].dt.dayofyear >= 182]

    if len(spring_frosts) > 0:
        last_spring = spring_frosts["date"].max()
        last_spring_frosts.append({
            "year"        : year,
            "date"        : last_spring,
            "day_of_year" : last_spring.dayofyear,
            "month_day"   : last_spring.strftime("%b %d")
        })

    if len(autumn_frosts) > 0:
        first_autumn = autumn_frosts["date"].min()
        first_autumn_frosts.append({
            "year"        : year,
            "date"        : first_autumn,
            "day_of_year" : first_autumn.dayofyear,
            "month_day"   : first_autumn.strftime("%b %d")
        })

spring_df = pd.DataFrame(last_spring_frosts)
autumn_df = pd.DataFrame(first_autumn_frosts)

print("--- LAST SPRING FROST ---")
print(f"Average last spring frost:  day "
      f"{spring_df['day_of_year'].mean():.0f} of year = around "
      f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(spring_df['day_of_year'].mean())-1):%B %d}")
print(f"Earliest last spring frost: "
      f"{spring_df.loc[spring_df['day_of_year'].idxmin(), 'month_day']} "
      f"({spring_df.loc[spring_df['day_of_year'].idxmin(), 'year']:.0f})")
print(f"Latest last spring frost:   "
      f"{spring_df.loc[spring_df['day_of_year'].idxmax(), 'month_day']} "
      f"({spring_df.loc[spring_df['day_of_year'].idxmax(), 'year']:.0f})")

p10 = spring_df["day_of_year"].quantile(0.10)
p50 = spring_df["day_of_year"].quantile(0.50)
p90 = spring_df["day_of_year"].quantile(0.90)

print(f"\n10th percentile: day {p10:.0f} — "
      f"In 90% of years frost still occurs after this date")
print(f"50th percentile: day {p50:.0f} — "
      f"In 50% of years the last frost is around this date")
print(f"90th percentile: day {p90:.0f} — "
      f"In only 10% of years does frost occur after this date")

print("\n--- FIRST AUTUMN FROST ---")
print(f"Average first autumn frost: day "
      f"{autumn_df['day_of_year'].mean():.0f} of year = around "
      f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(autumn_df['day_of_year'].mean())-1):%B %d}")

print("\n--- FROST-FREE GROWING SEASON ---")
season_lengths = (autumn_df["day_of_year"].values -
                  spring_df["day_of_year"].values[:len(autumn_df)])
print(f"Average growing season length: {season_lengths.mean():.0f} days")
print(f"Shortest growing season:       {season_lengths.min():.0f} days")
print(f"Longest growing season:        {season_lengths.max():.0f} days")

# -------------------------------------------------------
# STEP 7 - HEAT DAY ANALYSIS
# -------------------------------------------------------

print("\n\n=== HEAT DAY ANALYSIS ===\n")

heat_days    = df_complete[df_complete["temp_max"] >= HEAT_STRESS]
extreme_days = df_complete[df_complete["temp_max"] >= EXTREME_HEAT]

print(f"Days above {HEAT_STRESS}°C per year: "
      f"{len(heat_days)/n_complete_years:.1f}")
print(f"Days above {EXTREME_HEAT}°C per year: "
      f"{len(extreme_days)/n_complete_years:.1f}")

print(f"\n--- HEAT DAYS BY MONTH (above {HEAT_STRESS}°C) ---\n")

monthly_heat = (df_complete[df_complete["temp_max"] >= HEAT_STRESS]
                .groupby("month").size() / n_complete_years)
monthly_heat.index = monthly_heat.index.map(month_names)

for month, count in monthly_heat.items():
    bar = "█" * int(count)
    print(f"{month:<12} {count:>5.1f} days   {bar}")

# -------------------------------------------------------
# STEP 8 - RAINFALL PATTERNS
# -------------------------------------------------------

print("\n\n=== RAINFALL PATTERNS ===\n")

monthly_rain = df_complete.groupby("month").agg(
    avg_monthly_rain = ("precipitation", "sum"),
    rainy_days       = ("precipitation", lambda x: (x > 1.0).sum()),
).round(1)

monthly_rain["avg_monthly_rain"] = (
    monthly_rain["avg_monthly_rain"] / n_complete_years).round(1)
monthly_rain["avg_rainy_days"] = (
    monthly_rain["rainy_days"] / n_complete_years).round(1)

monthly_rain.index = monthly_rain.index.map(month_names)

print(f"{'Month':<12} {'Avg Rain(mm)':>14} {'Rainy Days':>12}")
print("-" * 42)
for month, row in monthly_rain.iterrows():
    print(f"{month:<12} {row['avg_monthly_rain']:>14.1f} "
          f"{row['avg_rainy_days']:>12.1f}")

total_annual = monthly_rain["avg_monthly_rain"].sum()
print(f"\nAverage annual rainfall: {total_annual:.1f} mm")

# -------------------------------------------------------
# SAVE PHASE 6 OUTPUTS
# -------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

monthly_temp.to_csv("data/processed/monthly_temperature_stats.csv")
monthly_rain.to_csv("data/processed/monthly_rainfall_stats.csv")
spring_df.to_csv("data/processed/last_spring_frosts.csv", index=False)

print("\nSummary statistics saved to data/processed/")
print("\n=== PHASE 6 ANALYSIS COMPLETE ===")

# -------------------------------------------------------
# PHASE 7 - CLIMATE TREND ANALYSIS
# -------------------------------------------------------
# From here we use df_complete only (no partial years).
# This ensures annual statistics are never distorted by
# years that only have a few months of data.

print("\n\n" + "=" * 60)
print("PHASE 7 - CLIMATE TREND ANALYSIS")
print("=" * 60)

# -------------------------------------------------------
# STEP 9 - ANNUAL AVERAGE TEMPERATURES
# -------------------------------------------------------

annual_temp = df_complete.groupby("year").agg(
    avg_temp   = ("temp_avg",      "mean"),
    avg_max    = ("temp_max",      "mean"),
    avg_min    = ("temp_min",      "mean"),
    total_rain = ("precipitation", "sum"),
    frost_days = ("temp_min",      lambda x: (x <= 0).sum()),
    heat_days  = ("temp_max",      lambda x: (x >= 35).sum()),
).round(3)

print("\n--- ANNUAL AVERAGES (first and last 5 years) ---\n")
print(annual_temp.head())
print("...")
print(annual_temp.tail())

# -------------------------------------------------------
# STEP 10 - TEMPERATURE TREND: FULL PERIOD
# -------------------------------------------------------

print("\n\n--- TEMPERATURE TREND (full period) ---\n")

x = annual_temp.index.values
y = annual_temp["avg_temp"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

warming_per_decade = slope * 10
total_years        = x.max() - x.min()

print(f"Temperature trend: {slope:.4f}°C per year")
print(f"                   {warming_per_decade:.3f}°C per decade")
print(f"R-squared:         {r_value**2:.3f}")
print(f"P-value:           {p_value:.4f}")
print(f"Standard error:    {std_err:.4f}")

print("\n--- WHAT THESE NUMBERS MEAN ---\n")

print(f"Slope: {slope:.4f}°C per year")
print(f"  → Each year the average temperature has shifted")
print(f"    by approximately {slope:.4f}°C")
print(f"  → Over 10 years that is {warming_per_decade:.2f}°C")
print(f"  → Over {total_years} years: {slope*total_years:.2f}°C total shift")

print(f"\nR-squared: {r_value**2:.3f}")
print(f"  → {r_value**2*100:.1f}% of year-to-year temperature")
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

# -------------------------------------------------------
# STEP 11 - THREE-PERIOD COMPARISON
# -------------------------------------------------------

print("\n\n--- THREE-PERIOD COMPARISON ---\n")
print("Comparing early, middle, and recent climate periods.\n")

periods = {
    "Early  (1981-1995)" : (1981, 1995),
    "Middle (1996-2010)" : (1996, 2010),
    "Recent (2011-2025)" : (2011, 2025),
}

period_stats = {}

for period_name, (start, end) in periods.items():
    period_data = df_complete[
        (df_complete["year"] >= start) &
        (df_complete["year"] <= end)
    ]

    n_period_years = len(period_data["year"].unique())

    stats_dict = {
        "avg_temp"   : period_data["temp_avg"].mean(),
        "avg_max"    : period_data["temp_max"].mean(),
        "avg_min"    : period_data["temp_min"].mean(),
        "total_rain" : period_data["precipitation"].sum() / n_period_years,
        "frost_days" : (period_data["temp_min"] <= 0).sum() / n_period_years,
        "heat_days"  : (period_data["temp_max"] >= 35).sum() / n_period_years,
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

early  = period_stats["Early  (1981-1995)"]
recent = period_stats["Recent (2011-2025)"]

print("--- CHANGE FROM EARLY TO RECENT PERIOD ---\n")
print(f"Average temperature:  "
      f"{early['avg_temp']:.2f}°C → {recent['avg_temp']:.2f}°C   "
      f"({recent['avg_temp']-early['avg_temp']:+.2f}°C)")
print(f"Average maximum:      "
      f"{early['avg_max']:.2f}°C → {recent['avg_max']:.2f}°C   "
      f"({recent['avg_max']-early['avg_max']:+.2f}°C)")
print(f"Average minimum:      "
      f"{early['avg_min']:.2f}°C → {recent['avg_min']:.2f}°C   "
      f"({recent['avg_min']-early['avg_min']:+.2f}°C)")
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
# STEP 12 - MONTHLY TREND ANALYSIS
# -------------------------------------------------------

print("\n\n--- WARMING TREND BY MONTH ---\n")
print("Which months are warming fastest?\n")
print(f"{'Month':<12} {'Trend':>10} {'Per Decade':>12} {'Confidence':>15}")
print("-" * 55)

month_trends = {}

for month_num in range(1, 13):
    month_name  = month_names[month_num]
    month_data  = (df_complete[df_complete["month"] == month_num]
                   .groupby("year")["temp_avg"].mean())

    x_m = month_data.index.values
    y_m = month_data.values

    s, i, r, p, se = stats.linregress(x_m, y_m)

    if p < 0.01:
        conf = "Very strong ✓✓"
    elif p < 0.05:
        conf = "Strong ✓"
    elif p < 0.10:
        conf = "Moderate ~"
    else:
        conf = "Weak ✗"

    month_trends[month_name] = {"slope": s, "p_value": p}

    print(f"{month_name:<12} {s:>+8.4f}°C/yr "
          f"{s*10:>+8.3f}°C/dec "
          f"{conf:>15}")

# -------------------------------------------------------
# STEP 13 - FROST DAY TREND
# -------------------------------------------------------

print("\n\n--- FROST DAY TREND ---\n")

frost_by_year = df_complete.groupby("year").apply(
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
# STEP 14 - HEAT DAY TREND
# -------------------------------------------------------

print("\n--- HEAT DAY TREND ---\n")

heat_by_year = df_complete.groupby("year").apply(
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
# SAVE PHASE 7 OUTPUTS
# -------------------------------------------------------

annual_temp.to_csv(
    "data/processed/annual_temperature_trends.csv")
frost_by_year.to_csv(
    "data/processed/frost_days_by_year.csv", index=False)
heat_by_year.to_csv(
    "data/processed/heat_days_by_year.csv", index=False)

print("\n\nTrend data saved to data/processed/")
print("\n=== PHASE 7 TREND ANALYSIS COMPLETE ===")
# src/visualization.py
#
# Phase 16 - Climate Visualization for Vojvodina, Serbia
#
# Creates charts that make our climate analysis immediately
# visible and understandable - for farmers, not just scientists.
#
# Charts we create:
#  1. Annual average temperature trend (1981-2025)
#  2. Monthly temperature ranges (box plots)
#  3. Three-period temperature comparison
#  4. Frost days per year - declining trend
#  5. Heat days per year - increasing trend
#  6. Monthly rainfall pattern
#  7. Last spring frost date over time
#  8. Annual rainfall totals
#  9. Temperature anomalies vs baseline
# 10. Climate summary dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

INPUT_FILE   = "data/processed/vojvodina_clean.csv"
OUTPUT_FOLDER = "data/processed/charts"

# Colors we will use consistently across all charts
# Using a warm/cool palette that matches the subject matter
COLOR_HOT     = "#d62728"   # red    - heat / warming
COLOR_COLD    = "#1f77b4"   # blue   - cold / frost
COLOR_RAIN    = "#2ca02c"   # green  - rainfall
COLOR_TREND   = "#ff7f0e"   # orange - trend lines
COLOR_EARLY   = "#aec7e8"   # light blue - early period
COLOR_MIDDLE  = "#ffbb78"   # light orange - middle period
COLOR_RECENT  = "#ff7f0e"   # orange - recent period
COLOR_NEUTRAL = "#7f7f7f"   # grey   - neutral elements

# Chart style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"]      = 120
plt.rcParams["savefig.dpi"]     = 150
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#f8f8f8"

# Month names for labels
MONTH_NAMES_SHORT = [
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
]

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

# Complete years only (exclude partial years like 2026)
days_per_year  = df.groupby("year").size()
complete_years = days_per_year[days_per_year >= 350].index
df_complete    = df[df["year"].isin(complete_years)]

# Annual statistics
annual = df_complete.groupby("year").agg(
    avg_temp   = ("temp_avg",      "mean"),
    avg_max    = ("temp_max",      "mean"),
    avg_min    = ("temp_min",      "mean"),
    total_rain = ("precipitation", "sum"),
    frost_days = ("temp_min",      lambda x: (x <= 0).sum()),
    heat_days  = ("temp_max",      lambda x: (x >= 35).sum()),
).round(3)

# Last spring frost dates
spring_frost_file = "data/processed/last_spring_frosts.csv"
spring_df = pd.read_csv(spring_frost_file)
spring_df["date"] = pd.to_datetime(spring_df["date"])

# Make output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Data loaded: {len(df_complete):,} days "
      f"({df_complete['year'].min()}-{df_complete['year'].max()})")
print(f"Charts will be saved to: {OUTPUT_FOLDER}")
print("-" * 60)

# -------------------------------------------------------
# HELPER FUNCTION - saves and closes each chart
# -------------------------------------------------------

def save_chart(filename, title=""):
    """Save chart to file and close it."""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# -------------------------------------------------------
# CHART 1 - ANNUAL TEMPERATURE TREND
# -------------------------------------------------------
# Shows every year's average temperature as a dot,
# with a trend line showing the direction of change.

print("\nCreating Chart 1: Annual temperature trend...")

fig, ax = plt.subplots(figsize=(14, 6))

years     = annual.index.values
avg_temps = annual["avg_temp"].values

# Scatter plot - one dot per year
ax.scatter(years, avg_temps,
           color=COLOR_HOT, alpha=0.7,
           s=50, zorder=3, label="Annual average temperature")

# Calculate and draw trend line
slope, intercept, r, p, se = stats.linregress(years, avg_temps)
trend_line = slope * years + intercept
ax.plot(years, trend_line,
        color=COLOR_TREND, linewidth=2.5,
        linestyle="--", zorder=2,
        label=f"Trend: +{slope*10:.2f}°C per decade")

# Add a 10-year rolling average to show the curve
annual_series = pd.Series(avg_temps, index=years)
rolling_avg   = annual_series.rolling(window=10, center=True).mean()
ax.plot(years, rolling_avg.values,
        color="#9467bd", linewidth=2,
        zorder=2, label="10-year rolling average")

# Shade the three periods with different background colors
ax.axvspan(1981, 1995, alpha=0.08, color=COLOR_COLD,  label="Early period (1981-1995)")
ax.axvspan(1996, 2010, alpha=0.08, color=COLOR_NEUTRAL)
ax.axvspan(2011, 2025, alpha=0.08, color=COLOR_HOT,   label="Recent period (2011-2025)")

# Labels and formatting
ax.set_title("Annual Average Temperature — Vojvodina, Serbia (1981–2025)\n"
             f"Trend: +{slope*10:.2f}°C per decade  |  "
             f"Total shift: +{slope*(years.max()-years.min()):.2f}°C  |  "
             f"P-value: <0.0001",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average Temperature (°C)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.set_xlim(1980, 2026)

# Add period average labels
early_avg  = df_complete[df_complete["year"].between(1981,1995)]["temp_avg"].mean()
recent_avg = df_complete[df_complete["year"].between(2011,2025)]["temp_avg"].mean()
ax.annotate(f"Early avg\n{early_avg:.1f}°C",
            xy=(1988, early_avg), fontsize=9,
            color=COLOR_COLD, fontweight="bold",
            ha="center")
ax.annotate(f"Recent avg\n{recent_avg:.1f}°C",
            xy=(2018, recent_avg+0.4), fontsize=9,
            color=COLOR_HOT, fontweight="bold",
            ha="center")

save_chart("01_annual_temperature_trend.png")

# -------------------------------------------------------
# CHART 2 - MONTHLY TEMPERATURE DISTRIBUTIONS (BOX PLOTS)
# -------------------------------------------------------
# A box plot shows the full range of temperatures for
# each month - not just the average.
# The box = middle 50% of observations
# The line in the middle = median
# The whiskers = typical range
# The dots = extreme outliers

print("Creating Chart 2: Monthly temperature distributions...")

fig, ax = plt.subplots(figsize=(14, 7))

# Build monthly data for box plots
monthly_data = []
for month in range(1, 13):
    monthly_data.append(
        df_complete[df_complete["month"] == month]["temp_avg"].values
    )

# Draw box plots
bp = ax.boxplot(monthly_data,
                patch_artist=True,  # filled boxes
                notch=False,
                medianprops=dict(color="black", linewidth=2))

# Color boxes by temperature - cool to warm
colors = [
    "#313695","#4575b4","#74add1",  # Jan Feb Mar - blue
    "#abd9e9","#e0f3f8","#fee090",  # Apr May Jun - light
    "#fdae61","#f46d43","#d73027",  # Jul Aug Sep - orange/red
    "#a50026","#4575b4","#313695"   # Oct Nov Dec
]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add a horizontal line at 0°C (freezing point)
ax.axhline(y=0, color=COLOR_COLD, linestyle="--",
           linewidth=1.5, alpha=0.7, label="Freezing point (0°C)")

# Add heat stress line
ax.axhline(y=35, color=COLOR_HOT, linestyle="--",
           linewidth=1.5, alpha=0.7, label="Heat stress threshold (35°C)")

ax.set_title("Monthly Temperature Distribution — Vojvodina, Serbia (1981–2025)\n"
             "Box shows middle 50% of daily averages  |  "
             "Line = median  |  Dots = extremes",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Daily Average Temperature (°C)", fontsize=12)
ax.set_xticklabels(MONTH_NAMES_SHORT, fontsize=11)
ax.legend(fontsize=10)

save_chart("02_monthly_temperature_distributions.png")

# -------------------------------------------------------
# CHART 3 - THREE-PERIOD COMPARISON BAR CHART
# -------------------------------------------------------
# Side-by-side bars showing how the climate has changed
# across three 15-year periods.

print("Creating Chart 3: Three-period temperature comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

periods = {
    "Early\n(1981–1995)" : (1981, 1995),
    "Middle\n(1996–2010)": (1996, 2010),
    "Recent\n(2011–2025)": (2011, 2025),
}
period_colors = [COLOR_EARLY, COLOR_MIDDLE, COLOR_RECENT]
period_labels = list(periods.keys())

# Panel 1 - Average temperature
avg_temps_periods = []
for (start, end) in periods.values():
    val = df_complete[df_complete["year"].between(start,end)]["temp_avg"].mean()
    avg_temps_periods.append(val)

bars = axes[0].bar(period_labels, avg_temps_periods,
                   color=period_colors, edgecolor="black", linewidth=0.8)
axes[0].set_title("Average Temperature\n(°C)", fontweight="bold")
axes[0].set_ylim(10, 14.5)
for bar, val in zip(bars, avg_temps_periods):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{val:.2f}°C", ha="center", fontweight="bold", fontsize=11)

# Panel 2 - Frost days per year
frost_periods = []
for (start, end) in periods.values():
    period_data = df_complete[df_complete["year"].between(start,end)]
    n_yrs = len(period_data["year"].unique())
    val   = (period_data["temp_min"] <= 0).sum() / n_yrs
    frost_periods.append(val)

bars = axes[1].bar(period_labels, frost_periods,
                   color=period_colors, edgecolor="black", linewidth=0.8)
axes[1].set_title("Frost Days per Year\n(days with min temp ≤ 0°C)",
                  fontweight="bold")
axes[1].set_ylim(0, 120)
for bar, val in zip(bars, frost_periods):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)

# Panel 3 - Heat days per year
heat_periods = []
for (start, end) in periods.values():
    period_data = df_complete[df_complete["year"].between(start,end)]
    n_yrs = len(period_data["year"].unique())
    val   = (period_data["temp_max"] >= 35).sum() / n_yrs
    heat_periods.append(val)

bars = axes[2].bar(period_labels, heat_periods,
                   color=period_colors, edgecolor="black", linewidth=0.8)
axes[2].set_title("Heat Days per Year\n(days with max temp ≥ 35°C)",
                  fontweight="bold")
axes[2].set_ylim(0, 30)
for bar, val in zip(bars, heat_periods):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)

fig.suptitle("Climate Change Across Three Periods — Vojvodina, Serbia",
             fontsize=14, fontweight="bold", y=1.02)

save_chart("03_three_period_comparison.png")

# -------------------------------------------------------
# CHART 4 - FROST DAYS PER YEAR (DECLINING TREND)
# -------------------------------------------------------

print("Creating Chart 4: Frost days per year...")

fig, ax = plt.subplots(figsize=(14, 6))

frost_by_year = annual["frost_days"]

ax.bar(frost_by_year.index, frost_by_year.values,
       color=COLOR_COLD, alpha=0.7, label="Frost days per year")

# Trend line
s, i, r, p, se = stats.linregress(
    frost_by_year.index.values,
    frost_by_year.values
)
trend = s * frost_by_year.index.values + i
ax.plot(frost_by_year.index, trend,
        color=COLOR_HOT, linewidth=2.5,
        linestyle="--",
        label=f"Trend: {s*10:.1f} days per decade")

# Rolling average
rolling = frost_by_year.rolling(window=10, center=True).mean()
ax.plot(frost_by_year.index, rolling,
        color="#9467bd", linewidth=2,
        label="10-year rolling average")

ax.set_title("Frost Days per Year — Vojvodina, Serbia (1981–2025)\n"
             f"Trend: {s*10:.1f} fewer frost days per decade  |  "
             f"Total reduction: ~{abs(s*44):.0f} days since 1981  |  "
             f"P-value: <0.0001",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Frost Days (min temp ≤ 0°C)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(1980, 2026)

save_chart("04_frost_days_per_year.png")

# -------------------------------------------------------
# CHART 5 - HEAT DAYS PER YEAR (INCREASING TREND)
# -------------------------------------------------------

print("Creating Chart 5: Heat days per year...")

fig, ax = plt.subplots(figsize=(14, 6))

heat_by_year = annual["heat_days"]

# Color bars by intensity
bar_colors = [
    COLOR_HOT if v >= 20 else
    COLOR_TREND if v >= 10 else
    "#fdae61"
    for v in heat_by_year.values
]

ax.bar(heat_by_year.index, heat_by_year.values,
       color=bar_colors, alpha=0.8, label="Heat days per year")

# Trend line
s, i, r, p, se = stats.linregress(
    heat_by_year.index.values,
    heat_by_year.values
)
trend = s * heat_by_year.index.values + i
ax.plot(heat_by_year.index, trend,
        color="darkred", linewidth=2.5,
        linestyle="--",
        label=f"Trend: +{s*10:.1f} days per decade")

# Rolling average
rolling = heat_by_year.rolling(window=10, center=True).mean()
ax.plot(heat_by_year.index, rolling,
        color="#9467bd", linewidth=2,
        label="10-year rolling average")

# Highlight 2024 record
if 2024 in heat_by_year.index:
    ax.annotate(f"2024 record\n{int(heat_by_year[2024])} days",
                xy=(2024, heat_by_year[2024]),
                xytext=(2019, heat_by_year[2024]+3),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                fontsize=10, color="darkred", fontweight="bold")

ax.set_title("Heat Days per Year (max temp ≥ 35°C) — Vojvodina, Serbia (1981–2025)\n"
             f"Trend: +{s*10:.1f} heat days per decade  |  "
             f"Early period avg: 7.9/yr  |  "
             f"Recent period avg: 20.5/yr",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Heat Days (max temp ≥ 35°C)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(1980, 2026)

# Add colored legend patches
early_patch  = mpatches.Patch(color="#fdae61", alpha=0.8,
                               label="Below 10 days (low stress)")
mid_patch    = mpatches.Patch(color=COLOR_TREND, alpha=0.8,
                               label="10–19 days (moderate stress)")
recent_patch = mpatches.Patch(color=COLOR_HOT, alpha=0.8,
                               label="20+ days (high stress)")
ax.legend(handles=[early_patch, mid_patch, recent_patch],
          loc="upper left", fontsize=9)

save_chart("05_heat_days_per_year.png")

# -------------------------------------------------------
# CHART 6 - MONTHLY RAINFALL PATTERN
# -------------------------------------------------------

print("Creating Chart 6: Monthly rainfall pattern...")

fig, ax = plt.subplots(figsize=(12, 6))

monthly_rain = df_complete.groupby("month").agg(
    avg_rain   = ("precipitation", lambda x: x.sum() / len(df_complete["year"].unique())),
    rainy_days = ("precipitation", lambda x: (x > 1.0).sum() / len(df_complete["year"].unique())),
).round(1)

bars = ax.bar(range(1, 13), monthly_rain["avg_rain"].values,
              color=COLOR_RAIN, alpha=0.75,
              edgecolor="darkgreen", linewidth=0.8,
              label="Average monthly rainfall (mm)")

# Add rainy day count on top of each bar
for i, (rain, rdays) in enumerate(zip(monthly_rain["avg_rain"].values,
                                       monthly_rain["rainy_days"].values)):
    ax.text(i+1, rain+1, f"{rdays:.0f}d",
            ha="center", va="bottom", fontsize=9,
            color="darkgreen", fontweight="bold")

ax.set_title("Average Monthly Rainfall — Vojvodina, Serbia (1981–2025)\n"
             "Numbers above bars = average rainy days per month (>1mm)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Average Rainfall (mm)", fontsize=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_NAMES_SHORT, fontsize=11)

# Shade the growing season
ax.axvspan(3.5, 9.5, alpha=0.08, color="green",
           label="Core growing season (Apr-Sep)")
ax.legend(fontsize=10)

# Add annual total annotation
annual_rain = monthly_rain["avg_rain"].sum()
ax.text(0.98, 0.95, f"Annual total: {annual_rain:.0f} mm",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

save_chart("06_monthly_rainfall.png")

# -------------------------------------------------------
# CHART 7 - LAST SPRING FROST DATE OVER TIME
# -------------------------------------------------------
# This chart shows whether spring frost is ending
# earlier as the climate warms - critical for planting.

print("Creating Chart 7: Last spring frost date over time...")

fig, ax = plt.subplots(figsize=(14, 6))

years_sf = spring_df["year"].values
doys_sf  = spring_df["day_of_year"].values  # day of year

ax.scatter(years_sf, doys_sf,
           color=COLOR_COLD, alpha=0.7, s=50, zorder=3,
           label="Last spring frost (day of year)")

# Trend line
s, i, r, p, se = stats.linregress(years_sf, doys_sf)
trend = s * years_sf + i
ax.plot(years_sf, trend,
        color=COLOR_HOT, linewidth=2.5,
        linestyle="--",
        label=f"Trend: {s*10:.1f} days per decade")

# Rolling average
doy_series = pd.Series(doys_sf, index=years_sf)
rolling    = doy_series.rolling(window=10, center=True).mean()
ax.plot(years_sf, rolling.values,
        color="#9467bd", linewidth=2,
        label="10-year rolling average")

# Add horizontal reference lines for key dates
key_dates = {
    "Mar 15 (day 74)" : 74,
    "Apr 1  (day 91)" : 91,
    "Apr 15 (day 105)": 105,
    "Apr 30 (day 120)": 120,
}
for label, doy in key_dates.items():
    ax.axhline(y=doy, color=COLOR_NEUTRAL,
               linestyle=":", alpha=0.6, linewidth=1)
    ax.text(1981.3, doy+0.5, label,
            fontsize=8, color=COLOR_NEUTRAL)

ax.set_title("Last Spring Frost Date per Year — Vojvodina, Serbia (1981–2025)\n"
             "Lower values = earlier end to frost season = earlier safe planting",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Day of Year (1=Jan 1, 91=Apr 1, 120=Apr 30)", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(1980, 2026)
ax.set_ylim(50, 135)

# Add p-value annotation
conf_text = f"P-value: {p:.4f}" if p >= 0.0001 else "P-value: <0.0001"
ax.text(0.02, 0.05, conf_text,
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

save_chart("07_last_spring_frost_trend.png")

# -------------------------------------------------------
# CHART 8 - ANNUAL RAINFALL TOTALS
# -------------------------------------------------------

print("Creating Chart 8: Annual rainfall totals...")

fig, ax = plt.subplots(figsize=(14, 6))

rain_by_year = annual["total_rain"]
avg_rain     = rain_by_year.mean()

# Color bars - blue if above average, brown if below
bar_colors = [
    COLOR_RAIN if v >= avg_rain else "#c49a3c"
    for v in rain_by_year.values
]

ax.bar(rain_by_year.index, rain_by_year.values,
       color=bar_colors, alpha=0.75, edgecolor="none")

# Average line
ax.axhline(y=avg_rain, color="navy", linewidth=2,
           linestyle="--",
           label=f"Long-term average: {avg_rain:.0f} mm")

# Rolling average
rolling = rain_by_year.rolling(window=10, center=True).mean()
ax.plot(rain_by_year.index, rolling,
        color="#9467bd", linewidth=2,
        label="10-year rolling average")

# Highlight notable drought years
drought_threshold = avg_rain * 0.85  # 15% below average
for year, rain in rain_by_year.items():
    if rain < drought_threshold:
        ax.text(year, rain - 15, str(year),
                ha="center", va="top", fontsize=7,
                color="#c49a3c", rotation=90)

ax.set_title("Annual Rainfall Totals — Vojvodina, Serbia (1981–2025)\n"
             "Blue = above average  |  Brown = below average  |  "
             f"Average: {avg_rain:.0f} mm/year",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Total Annual Rainfall (mm)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(1980, 2026)

blue_patch  = mpatches.Patch(color=COLOR_RAIN, alpha=0.75,
                              label="Above average rainfall")
brown_patch = mpatches.Patch(color="#c49a3c", alpha=0.75,
                              label="Below average rainfall (year labels = drought risk)")
ax.legend(handles=[blue_patch, brown_patch], fontsize=9)

save_chart("08_annual_rainfall.png")

# -------------------------------------------------------
# CHART 9 - TEMPERATURE ANOMALY
# -------------------------------------------------------
# Anomaly = how much warmer or cooler than the long-term average
# This is a standard climate science visualization.

print("Creating Chart 9: Temperature anomalies...")

fig, ax = plt.subplots(figsize=(14, 6))

# Baseline = average of the first 30 years (1981-2010)
baseline = df_complete[
    df_complete["year"].between(1981, 2010)
]["temp_avg"].mean()

anomalies = annual["avg_temp"] - baseline

# Color bars by positive/negative anomaly
bar_colors = [COLOR_HOT if v > 0 else COLOR_COLD
              for v in anomalies.values]

ax.bar(anomalies.index, anomalies.values,
       color=bar_colors, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=1.2)

ax.set_title(f"Annual Temperature Anomaly — Vojvodina, Serbia (1981–2025)\n"
             f"Baseline: 1981–2010 average ({baseline:.2f}°C)  |  "
             f"Red = warmer than baseline  |  Blue = cooler than baseline",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Temperature Anomaly (°C above/below baseline)", fontsize=12)
ax.set_xlim(1980, 2026)

# Add annotation for recent years
recent_anomaly = anomalies[anomalies.index >= 2020].mean()
ax.text(0.98, 0.95,
        f"2020–2025 avg anomaly: +{recent_anomaly:.2f}°C",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color=COLOR_HOT,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

save_chart("09_temperature_anomaly.png")

# -------------------------------------------------------
# CHART 10 - CLIMATE DASHBOARD
# -------------------------------------------------------
# A 2x3 summary panel combining key statistics in one view.

print("Creating Chart 10: Climate summary dashboard...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Climate Summary Dashboard — Vojvodina, Serbia (1981–2025)",
             fontsize=15, fontweight="bold", y=1.01)

# --- Panel 1: Annual temperature trend ---
ax = axes[0, 0]
s, i, r, p, se = stats.linregress(
    annual.index.values, annual["avg_temp"].values)
ax.scatter(annual.index, annual["avg_temp"],
           color=COLOR_HOT, alpha=0.6, s=30)
ax.plot(annual.index, s * annual.index.values + i,
        color=COLOR_TREND, linewidth=2, linestyle="--")
ax.set_title("Annual Average Temperature", fontweight="bold")
ax.set_ylabel("°C")
ax.set_xlabel("Year")

# --- Panel 2: Frost days ---
ax = axes[0, 1]
ax.bar(annual.index, annual["frost_days"],
       color=COLOR_COLD, alpha=0.7)
s2, i2, r2, p2, se2 = stats.linregress(
    annual.index.values, annual["frost_days"].values)
ax.plot(annual.index, s2 * annual.index.values + i2,
        color=COLOR_HOT, linewidth=2, linestyle="--")
ax.set_title("Frost Days per Year", fontweight="bold")
ax.set_ylabel("Days")
ax.set_xlabel("Year")

# --- Panel 3: Heat days ---
ax = axes[0, 2]
ax.bar(annual.index, annual["heat_days"],
       color=COLOR_HOT, alpha=0.7)
s3, i3, r3, p3, se3 = stats.linregress(
    annual.index.values, annual["heat_days"].values)
ax.plot(annual.index, s3 * annual.index.values + i3,
        color="darkred", linewidth=2, linestyle="--")
ax.set_title("Heat Days per Year (≥35°C)", fontweight="bold")
ax.set_ylabel("Days")
ax.set_xlabel("Year")

# --- Panel 4: Monthly temperature profile ---
ax = axes[1, 0]
monthly_avg = df_complete.groupby("month")["temp_avg"].mean()
monthly_max = df_complete.groupby("month")["temp_max"].mean()
monthly_min = df_complete.groupby("month")["temp_min"].mean()
months      = range(1, 13)
ax.fill_between(months, monthly_min.values,
                monthly_max.values,
                alpha=0.2, color=COLOR_HOT,
                label="Avg min–max range")
ax.plot(months, monthly_avg.values,
        color=COLOR_HOT, linewidth=2,
        marker="o", markersize=5,
        label="Avg temperature")
ax.axhline(y=0, color=COLOR_COLD, linestyle="--",
           linewidth=1, alpha=0.7)
ax.set_title("Monthly Temperature Profile", fontweight="bold")
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Month")
ax.set_xticks(months)
ax.set_xticklabels(MONTH_NAMES_SHORT, fontsize=8)
ax.legend(fontsize=8)

# --- Panel 5: Monthly rainfall ---
ax = axes[1, 1]
monthly_rain_avg = df_complete.groupby("month")["precipitation"].sum() / len(complete_years)
ax.bar(months, monthly_rain_avg.values,
       color=COLOR_RAIN, alpha=0.7)
ax.set_title("Average Monthly Rainfall", fontweight="bold")
ax.set_ylabel("Rainfall (mm)")
ax.set_xlabel("Month")
ax.set_xticks(list(months))
ax.set_xticklabels(MONTH_NAMES_SHORT, fontsize=8)

# --- Panel 6: Key statistics text summary ---
ax = axes[1, 2]
ax.axis("off")  # no axes - just text

early_avg  = df_complete[df_complete["year"].between(1981,1995)]["temp_avg"].mean()
recent_avg = df_complete[df_complete["year"].between(2011,2025)]["temp_avg"].mean()
warming    = recent_avg - early_avg

summary_text = (
    f"KEY CLIMATE STATISTICS\n"
    f"{'='*30}\n\n"
    f"Period: 1981–2025 (45 years)\n\n"
    f"TEMPERATURE\n"
    f"Overall average:  {df_complete['temp_avg'].mean():.1f}°C\n"
    f"Early period avg: {early_avg:.1f}°C\n"
    f"Recent period avg:{recent_avg:.1f}°C\n"
    f"Total warming:   +{warming:.1f}°C\n"
    f"Rate: +{s*10:.2f}°C per decade\n\n"
    f"FROST\n"
    f"Avg frost days/yr: {annual['frost_days'].mean():.0f}\n"
    f"Trend: {s2*10:.1f} days/decade\n"
    f"Avg last frost:  ~Apr 05\n\n"
    f"HEAT\n"
    f"Avg heat days/yr:  {annual['heat_days'].mean():.0f}\n"
    f"Trend: +{s3*10:.1f} days/decade\n"
    f"Record: 48 days (2024)\n\n"
    f"RAINFALL\n"
    f"Annual average: {annual['total_rain'].mean():.0f} mm\n"
    f"Wettest month:  June (75mm)\n"
    f"Driest month:   Feb (38mm)"
)

ax.text(0.05, 0.95, summary_text,
        transform=ax.transAxes,
        fontsize=9.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0",
                  alpha=0.8, edgecolor="grey"))

save_chart("10_climate_dashboard.png")

# -------------------------------------------------------
# DONE
# -------------------------------------------------------

print("\n" + "=" * 60)
print("ALL CHARTS CREATED SUCCESSFULLY")
print("=" * 60)
print(f"\nCharts saved to: {OUTPUT_FOLDER}")
print("\nFiles created:")
for f in sorted(os.listdir(OUTPUT_FOLDER)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(OUTPUT_FOLDER, f))
        print(f"  {f}  ({size//1024} KB)")
print("\nOpen the data/processed/charts/ folder to view your charts.")
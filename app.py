# app.py
import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")

from config import CONFIG
from crop_model import CROPS, get_crop, get_heat_stress_temp, get_frost_kill_temp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

STAT_FILE = CONFIG["risk_scores_file"]
ML_FILE   = CONFIG["ml_pred_file"]
LOCATION  = CONFIG["location_name"]
DATA_FROM = CONFIG["analysis_start"]
DATA_TO   = CONFIG["analysis_end"]

AUTUMN_MONTHS = {10, 11, 12}

OUTDOOR_DATES = {
    "potato"  : "Mar 10 – Mar 25",
    "tomato"  : "Apr 10 – Apr 20",
    "onion"   : "Mar 1 – Mar 20",
    "cucumber": "Apr 25 – May 5",
}

INDOOR_DATES = {
    "potato"  : "N/A (tubers)",
    "tomato"  : "Feb 15 – Mar 1",
    "onion"   : "Jan – Feb",
    "cucumber": "Apr 1–10 (opt.)",
}

HARVEST_DATES = {
    "potato"  : "Late May – June",
    "tomato"  : "July – September",
    "onion"   : "Late June – July",
    "cucumber": "June – early July",
}


# -------------------------------------------------------
# RISK HELPERS
# -------------------------------------------------------

def combine_stat_ml(stat_prob, ml_prob, ml_weight=0.55):
    if ml_prob is None or (isinstance(ml_prob, float)
                           and np.isnan(ml_prob)):
        return stat_prob
    return stat_prob * (1 - ml_weight) + ml_prob * ml_weight


def risk_label(score):
    if score < 0.15:   return "Very Low"
    elif score < 0.25: return "Low"
    elif score < 0.40: return "Moderate"
    elif score < 0.55: return "High"
    else:              return "Very High"


def risk_color(score):
    """Return a Rich color name based on risk score."""
    if score < 0.15:   return "bright_green"
    elif score < 0.25: return "green"
    elif score < 0.40: return "yellow"
    elif score < 0.55: return "orange1"
    else:              return "red"


def pct_colored(value):
    """Return a colored percentage string."""
    color = risk_color(value)
    return f"[{color}]{value*100:.0f}%[/{color}]"


def score_colored(value):
    """Return colored score with label."""
    color = risk_color(value)
    label = risk_label(value)
    return f"[{color}]{value*100:.0f}% {label}[/{color}]"


# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------

def load_data():
    if not os.path.exists(STAT_FILE):
        console.print("[red]ERROR: Run python src/risk_model.py first[/red]")
        sys.exit(1)
    stat_df = pd.read_csv(STAT_FILE)
    ml_df   = pd.read_csv(ML_FILE) \
        if os.path.exists(ML_FILE) else None
    return stat_df, ml_df


def get_predictions(crop_key, stat_df, ml_df):
    stat_crop = stat_df[stat_df["crop_key"] == crop_key].copy()
    ml_crop   = ml_df[ml_df["crop_key"] == crop_key].copy() \
        if ml_df is not None else None

    crop = get_crop(crop_key)
    rows = []

    for _, sr in stat_crop.iterrows():
        month      = sr["month"]
        day        = sr["day"]
        date_label = sr["date_label"]
        is_autumn  = month in AUTUMN_MONTHS

        if is_autumn and crop["season_type"] == "warm":
            continue

        ml_frost = ml_heat = ml_rain = np.nan
        if ml_crop is not None and not ml_crop.empty:
            m = ml_crop[ml_crop["date_label"] == date_label]
            if not m.empty:
                ml_frost = m["frost_kill_prob"].iloc[0] \
                    if "frost_kill_prob"   in m.columns else np.nan
                ml_heat  = m["heat_stress_prob"].iloc[0] \
                    if "heat_stress_prob"  in m.columns else np.nan
                ml_rain  = m["rain_deficit_prob"].iloc[0] \
                    if "rain_deficit_prob" in m.columns else np.nan

        winter_surv = np.nan
        if is_autumn and "winter_survival_prob" in sr.index:
            ws = sr.get("winter_survival_prob", np.nan)
            if not pd.isna(ws):
                frost_c     = 1.0 - ws
                winter_surv = ws * 100
            else:
                frost_c = combine_stat_ml(
                    sr["frost_kill_prob"], ml_frost
                )
        else:
            frost_c = combine_stat_ml(
                sr["frost_kill_prob"], ml_frost
            )

        heat_c = combine_stat_ml(sr["heat_severe_prob"],  ml_heat)
        rain_c = combine_stat_ml(sr["rain_deficit_prob"], ml_rain)

        w = ({"frost":0.35,"heat":0.30,"rain":0.35}
             if crop["season_type"] == "cool"
             else {"frost":0.30,"heat":0.28,"rain":0.42})

        combined = (frost_c * w["frost"] +
                    heat_c  * w["heat"]  +
                    rain_c  * w["rain"])
        combined += frost_c * rain_c * 0.15

               # Stricter frost veto for highly frost-sensitive crops
        # Cucumber and tomato are killed by any frost
        # so we penalise frost risk more aggressively
        veto_threshold = 0.05 \
            if crop_key in {"cucumber", "tomato"} \
            else 0.10
        penalty_multiplier = 0.70 \
            if crop_key in {"cucumber", "tomato"} \
            else 0.50

        if (not is_autumn and
                crop["season_type"] == "warm" and
                frost_c > veto_threshold):
            combined += (
                (frost_c - veto_threshold) * penalty_multiplier
            )

        combined = min(1.0, max(0.0, combined))

        rows.append({
            "date_label" : date_label,
            "month"      : month,
            "is_autumn"  : is_autumn,
            "frost"      : frost_c,
            "heat"       : heat_c,
            "rain"       : rain_c,
            "combined"   : combined,
            "rating"     : risk_label(combined),
            "winter_surv": winter_surv,
        })

    return pd.DataFrame(rows).sort_values(
        "combined"
    ).reset_index(drop=True)


# -------------------------------------------------------
# PRINT SINGLE CROP
# -------------------------------------------------------

def print_crop(crop_key, stat_df, ml_df):
    crop      = get_crop(crop_key)
    preds     = get_predictions(crop_key, stat_df, ml_df)
    spring    = preds[~preds["is_autumn"]]
    autumn    = preds[preds["is_autumn"]]

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold white]{crop['name'].upper()}[/bold white]  "
        f"[dim]{crop['latin']}[/dim]\n"
        f"Season: [cyan]{crop['season_type']}[/cyan]  |  "
        f"Frost kills: [blue]{get_frost_kill_temp(crop_key)}°C[/blue]  |  "
        f"Heat stress: [red]{get_heat_stress_temp(crop_key)}°C[/red]  |  "
        f"Water: {crop['water']['total_season_mm']}mm  |  "
        f"Drought: {crop['water']['drought_sensitivity']}",
        box=box.ROUNDED,
        style="bold cyan"
    ))

    # Spring table
    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title="[bold]Spring Planting Risk[/bold]",
        title_style="cyan",
    )
    t.add_column("Date",         style="white",  width=10)
    t.add_column("Frost Risk",   justify="right", width=12)
    t.add_column("Heat Risk",    justify="right", width=10)
    t.add_column("Rain Deficit", justify="right", width=12)
    t.add_column("Score",        justify="right", width=18)

    best_idx = spring.index[0] if not spring.empty else -1

    for idx, r in spring.iterrows():
        date_str = (f"[bold yellow]▶ {r['date_label']}[/bold yellow]"
                    if idx == best_idx
                    else r['date_label'])
        t.add_row(
            date_str,
            pct_colored(r["frost"]),
            pct_colored(r["heat"]),
            pct_colored(r["rain"]),
            score_colored(r["combined"]),
        )

    console.print(t)

    # Autumn table (cool-season crops only)
    if not autumn.empty:
        ta = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            title="[bold]Autumn/Winter Planting Risk[/bold]",
            title_style="blue",
        )
        ta.add_column("Date",         style="white",  width=10)
        ta.add_column("Winter Kill",  justify="right", width=12)
        ta.add_column("Heat Risk",    justify="right", width=10)
        ta.add_column("Rain Deficit", justify="right", width=12)
        ta.add_column("Score",        justify="right", width=18)
        ta.add_column("Survival",     justify="right", width=14)

        for _, r in autumn.iterrows():
            surv_str = (
                f"[green]{r['winter_surv']:.0f}%[/green]"
                if not pd.isna(r["winter_surv"]) and
                r["winter_surv"] >= 75
                else (
                    f"[yellow]{r['winter_surv']:.0f}%[/yellow]"
                    if not pd.isna(r["winter_surv"]) and
                    r["winter_surv"] >= 50
                    else (
                        f"[red]{r['winter_surv']:.0f}%[/red]"
                        if not pd.isna(r["winter_surv"])
                        else "N/A"
                    )
                )
            )
            ta.add_row(
                r["date_label"],
                pct_colored(r["frost"]),
                pct_colored(r["heat"]),
                pct_colored(r["rain"]),
                score_colored(r["combined"]),
                surv_str,
            )

        console.print(ta)

    # Top 3 spring recommendations
    if not spring.empty:
        t3 = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            title="[bold]Top 3 Recommended Dates[/bold]",
            title_style="green",
        )
        t3.add_column("Rank",   width=5)
        t3.add_column("Date",   width=10)
        t3.add_column("Score",  justify="right", width=18)
        t3.add_column("Frost",  justify="right", width=8)
        t3.add_column("Heat",   justify="right", width=8)
        t3.add_column("Rain",   justify="right", width=8)

        labels = ["🥇 Best", "🥈 Good", "🥉 OK"]
        for rank, (_, r) in enumerate(spring.head(3).iterrows()):
            t3.add_row(
                labels[rank],
                f"[bold]{r['date_label']}[/bold]",
                score_colored(r["combined"]),
                pct_colored(r["frost"]),
                pct_colored(r["heat"]),
                pct_colored(r["rain"]),
            )

        console.print(t3)


# -------------------------------------------------------
# PRINT SUMMARY
# -------------------------------------------------------

def print_summary(stat_df, ml_df):
    console.print()
    console.print(Panel(
        f"[bold white]PLANTING ADVISOR SUMMARY[/bold white]\n"
        f"[dim]{LOCATION}  |  "
        f"{DATA_FROM}–{DATA_TO}  |  "
        f"No irrigation  |  NASA POWER + ML[/dim]",
        box=box.DOUBLE,
        style="bold cyan"
    ))

    # Overall risk summary
    rows = []
    for crop_key in CROPS:
        preds  = get_predictions(crop_key, stat_df, ml_df)
        spring = preds[~preds["is_autumn"]]
        if spring.empty:
            continue
        best = spring.iloc[0]
        rows.append({
            "crop"  : CROPS[crop_key]["name"],
            "key"   : crop_key,
            "date"  : best["date_label"],
            "frost" : best["frost"],
            "heat"  : best["heat"],
            "rain"  : best["rain"],
            "score" : best["combined"],
        })

    rows.sort(key=lambda x: x["score"])

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title="[bold]Risk Summary — Best Date Per Crop[/bold]",
        title_style="cyan",
    )
    t.add_column("Crop",       style="white bold", width=12)
    t.add_column("Best Date",  width=11)
    t.add_column("Frost",      justify="right", width=8)
    t.add_column("Heat",       justify="right", width=8)
    t.add_column("Rain",       justify="right", width=8)
    t.add_column("Score",      justify="right", width=18)

    for r in rows:
        t.add_row(
            r["crop"],
            r["date"],
            pct_colored(r["frost"]),
            pct_colored(r["heat"]),
            pct_colored(r["rain"]),
            score_colored(r["score"]),
        )

    console.print(t)

    # Planting calendar
    tc = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title="[bold]Planting Calendar[/bold]",
        title_style="cyan",
    )
    tc.add_column("Crop",     style="white bold", width=12)
    tc.add_column("Indoors",  width=18)
    tc.add_column("Outdoors", width=18)
    tc.add_column("Harvest",  width=18)

    for crop_key in CROPS:
        name = CROPS[crop_key]["name"]
        tc.add_row(
            name,
            INDOOR_DATES.get(crop_key, "N/A"),
            f"[bold]{OUTDOOR_DATES.get(crop_key, 'N/A')}[/bold]",
            HARVEST_DATES.get(crop_key, "N/A"),
        )

    console.print(tc)

    # Soil temperature
    rfile = "data/processed/soil_readiness_dates.csv"
    if os.path.exists(rfile):
        rdf = pd.read_csv(rfile)

        def doy_str(doy):
            return (pd.Timestamp("2024-01-01") +
                    pd.Timedelta(days=int(doy)-1)
                    ).strftime("%b %d")

        ts = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            title="[bold]Soil Temperature Readiness (0-10cm)[/bold]",
            title_style="cyan",
        )
        ts.add_column("Crop",      style="white bold", width=12)
        ts.add_column("Min Soil",  justify="right", width=10)
        ts.add_column("Avg Ready", justify="right", width=12)
        ts.add_column("Safe 90%",  justify="right", width=12)

        for crop_key in CROPS:
            cd = rdf[rdf["crop_key"] == crop_key]
            if cd.empty:
                continue
            mn  = cd["min_soil_c"].iloc[0]
            avg = doy_str(cd["day_of_year"].mean())
            p90 = doy_str(cd["day_of_year"].quantile(0.90))
            ts.add_row(
                CROPS[crop_key]["name"],
                f"{mn:.0f}°C",
                avg,
                p90,
            )

        console.print(ts)

    # Footer
    console.print()
    console.print(Panel(
        "[dim]Historical probabilities based on 2000–2026 NASA data.\n"
        "Not a weather forecast. Always check local conditions.\n"
        f"Vojvodina warming: +2.73°C since 1981. "
        f"Summer drought risk increasing.[/dim]",
        box=box.ROUNDED,
        style="dim"
    ))


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    stat_df, ml_df = load_data()
    requested = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if requested:
        if requested not in CROPS:
            console.print(
                f"[red]Unknown crop. "
                f"Available: {list(CROPS.keys())}[/red]"
            )
            sys.exit(1)
        print_crop(requested, stat_df, ml_df)
    else:
        for crop_key in CROPS:
            print_crop(crop_key, stat_df, ml_df)
        print_summary(stat_df, ml_df)


if __name__ == "__main__":
    main()
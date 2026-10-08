"""
precompute_climatology.py

Builds the small lookup table the Streamlit app reads, so the app never has
to touch the full historical CSV (or make a live API call) when someone
moves the date slider.

For each location, and for each day of the year, this looks at every
historical record within +/- 7 days of that date (across all ~86 years)
and summarises what "typical" looks like. It also works out, per location,
where a day's rain / wind / cloud / sunshine sits on a simple 1-5 "how much
of this is there" scale, using that location's own historical spread
(quintiles) as the ruler.

This is the placeholder/baseline layer described in the project notes:
it is NOT Adem's trained model. It exists so the app is fully working and
demoable on its own, and so the matching/scoring logic in app.py can be
built and tested right now. When Adem's model is ready, only the
`get_conditions()` function in app.py needs to change -- everything below
it (bucketing, matching, scoring, traffic light) stays exactly the same.

Run it once:
    python precompute_climatology.py full_weather_data.csv climatology.csv
"""

import sys
import pandas as pd
import numpy as np

USECOLS = [
    "date_UTC", "location",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "rain_sum", "wind_speed_10m_max", "cloud_cover_mean", "sunshine_duration",
]

# The four dimensions the app matches against user preference (1-5 scale).
# Temperature is deliberately NOT one of these -- see app.py for why.
INTENSITY_COLS = ["rain_sum", "wind_speed_10m_max", "cloud_cover_mean", "sunshine_duration"]

WINDOW_DAYS = 7  # +/- days around a target date-of-year, pooled across all years


def month_day_key(dt_index):
    """Stable month-day key that doesn't shift with leap years.
    29 Feb is folded into 28 Feb so every year lines up the same way."""
    md = dt_index.dt.strftime("%m-%d")
    return md.where(md != "02-29", "02-28")


def circular_window_mask(all_doy, target_doy, window):
    """True for rows within `window` days of target_doy, wrapping across
    the Dec 31 -> Jan 1 boundary. `all_doy` is precomputed once per
    location so this stays cheap inside the per-day loop."""
    diff = (all_doy - target_doy) % 365
    diff = np.minimum(diff, 365 - diff)
    return diff <= window


def main(src_path, out_path):
    print(f"Reading {src_path} ...")
    df = pd.read_csv(src_path, usecols=USECOLS, parse_dates=["date_UTC"])
    df["date_NZDT"] = (df["date_UTC"] + pd.Timedelta(hours=13)).dt.tz_localize(None).dt.normalize()
    df["month_day"] = month_day_key(df["date_NZDT"])
    df = df.dropna(subset=INTENSITY_COLS, how="all")

    locations = sorted(df["location"].unique())
    print("Locations:", locations)

    # Step 1: per-location quintile cut points for each intensity dimension.
    # These turn a raw value (e.g. "14.2 km/h wind") into a 1-5 bucket that
    # means "how much wind is this, for this city specifically".
    cutpoints = {}
    for loc in locations:
        loc_df = df[df["location"] == loc]
        cutpoints[loc] = {
            col: loc_df[col].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            for col in INTENSITY_COLS
        }

    def to_bucket(value, cuts):
        if pd.isna(value):
            return np.nan
        return int(np.searchsorted(cuts, value, side="right") + 1)  # 1..5

    # Step 2: for every location x calendar day, pool the +/- 7 day window
    # across all years and summarise it.
    all_month_days = sorted(df["month_day"].unique())
    target_doys = {
        md: pd.to_datetime(f"2001-{md}", format="%Y-%m-%d").dayofyear
        for md in all_month_days
    }
    rows = []
    for loc in locations:
        loc_df = df[df["location"] == loc]
        loc_doy = pd.to_datetime("2001-" + loc_df["month_day"], format="%Y-%m-%d").dt.dayofyear.to_numpy()
        for target_md in all_month_days:
            target_doy = target_doys[target_md]
            mask = circular_window_mask(loc_doy, target_doy, WINDOW_DAYS)
            window_df = loc_df[mask]
            if len(window_df) < 20:
                continue
            row = {"location": loc, "month_day": target_md, "n_days": len(window_df)}
            for col in ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]:
                row[f"{col}_typical"] = round(float(window_df[col].mean()), 1)
            for col in INTENSITY_COLS:
                typical = window_df[col].median()
                row[f"{col}_typical"] = round(float(typical), 2) if pd.notna(typical) else np.nan
                row[f"{col}_bucket"] = to_bucket(typical, cutpoints[loc][col])
            row["dry_day_pct"] = round(float((window_df["rain_sum"] < 1.0).mean() * 100), 1)
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["location", "month_day"])
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows ({len(locations)} locations x ~365 days)")

    # Quick sanity check in the spirit of "does this match what we already
    # know" -- Christchurch in January should look like summer, not winter.
    check = out[(out["location"] == "Christchurch") & (out["month_day"] == "01-15")]
    if not check.empty:
        print("\nSanity check -- Christchurch, 15 January:")
        print(check[["temperature_2m_mean_typical", "temperature_2m_max_typical",
                      "dry_day_pct", "sunshine_duration_bucket", "wind_speed_10m_max_bucket"]]
              .to_string(index=False))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "full_weather_data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "climatology.csv"
    main(src, out)

"""
GoodDay -- Outdoor Event Weather Suitability
Streamlit app (placeholder-model version)

WHAT THIS IS RIGHT NOW
-----------------------
This version answers "how suitable is this date?" using climatology.csv --
a lookup table built straight from ~86 years of history for four NZ cities
(see precompute_climatology.py). It is NOT Adem's trained model. It exists
so the app is fully working and demoable today, independent of where the
modelling work is at.

THE ONE SWAP-IN POINT
-----------------------
Every bit of matching/scoring/traffic-light logic below reads its input
from a single function: get_conditions(location, date). When Adem's model
is ready, that is the ONLY function that needs to change -- it can call his
model instead of the lookup table, as long as it returns the same shape of
dictionary. Nothing else in this file needs to be touched.

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GoodDay", page_icon="assets_placeholder", layout="centered")

LOCATIONS = ["Auckland", "Wellington", "Christchurch", "Dunedin"]

# The four things the user has an opinion about, and that we match on.
# Temperature is deliberately NOT one of these -- it's shown as supporting
# context instead (see the wireframe: "all the extra information").
DIMENSIONS = [
    ("sunshine_duration", "Sunshine"),
    ("rain_sum", "Rain"),
    ("cloud_cover_mean", "Cloud / overcast"),
    ("wind_speed_10m_max", "Wind"),
]

MATCH_LABELS = {0: ("Strong match", "#DFF3E6", "#1B6B3A"),
                1: ("Good match", "#E7F3E0", "#3A7A2E"),
                2: ("Average", "#FBECCB", "#8A6B1A"),
                3: ("Poor match", "#FBE2DE", "#A3362B")}

TRAFFIC = {"green": ("#279A5C", "Good"), "amber": ("#D99A1E", "Average"), "red": ("#C14A3C", "Poor")}


@st.cache_data
def load_climatology():
    return pd.read_csv("climatology.csv")


def month_day_key(date: dt.date) -> str:
    """Matches the key precompute_climatology.py uses -- 29 Feb folds to 28 Feb."""
    md = date.strftime("%m-%d")
    return "02-28" if md == "02-29" else md


def get_conditions(location: str, date: dt.date, clim: pd.DataFrame) -> dict | None:
    """
    *** THIS IS THE FUNCTION TO REPLACE WITH ADEM'S MODEL. ***

    Today: looks up the precomputed climatology row for this location and
    day-of-year (+/- 7 days, pooled across ~86 years) and returns it as a
    plain dict.

    Later: can call his trained model instead -- e.g. a Lambda endpoint,
    or a loaded pickle -- as long as it returns a dict with the same keys:
    one "<col>_bucket" (1-5) per dimension in DIMENSIONS, plus
    temperature_2m_mean_typical / _max_typical / _min_typical and
    dry_day_pct for the "extra information" panel.
    """
    md = month_day_key(date)
    row = clim[(clim["location"] == location) & (clim["month_day"] == md)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def match_level(predicted_bucket: float, preference: int) -> int:
    """0 = strong match, 1 = good, 2 = average, 3 = poor -- based on how far
    apart the predicted intensity and the stated preference are on the
    shared 1-5 scale. Not "low is always good" -- someone can prefer a 5."""
    if pd.isna(predicted_bucket):
        return 2  # not enough history for this exact date -- call it average, don't overclaim
    diff = abs(predicted_bucket - preference)
    return min(int(round(diff)), 3)


def render_badge(text: str, bg: str, fg: str):
    st.markdown(
        f'<span style="background:{bg};color:{fg};padding:4px 14px;border-radius:999px;'
        f'font-weight:600;font-size:0.9rem">{text}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- UI ----

st.title("GoodDay")
st.caption("Smarter date-picking for outdoor events, built on 85+ years of New Zealand weather data.")

clim = load_climatology()

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", LOCATIONS)
with col2:
    event_date = st.date_input("Event date", value=dt.date(2026, 12, 14))

st.subheader("Your preferences")
st.caption("1 = least preferred · 5 = preferred")

pref = {}
pcol1, pcol2, pcol3, pcol4 = st.columns(4)
for (key, label), col in zip(DIMENSIONS, [pcol1, pcol2, pcol3, pcol4]):
    with col:
        pref[key] = st.slider(label, 1, 5, 3, key=f"pref_{key}")

st.divider()

conditions = get_conditions(location, event_date, clim)

if conditions is None:
    st.warning("Not enough historical data for this exact date yet -- try a nearby date.")
else:
    st.subheader(f"Suitability for {location}, {event_date.strftime('%d %b')}")

    scores = []
    for key, label in DIMENSIONS:
        predicted_bucket = conditions.get(f"{key}_bucket")
        level = match_level(predicted_bucket, pref[key])
        scores.append(3 - level)  # 3=strong .. 0=poor, for the overall average below
        text, bg, fg = MATCH_LABELS[level]
        row_l, row_r = st.columns([1, 1])
        with row_l:
            st.markdown(f"**{label}**")
        with row_r:
            render_badge(text, bg, fg)

    overall_pct = round((sum(scores) / (len(scores) * 3)) * 100)
    if overall_pct >= 70:
        light = "green"
    elif overall_pct >= 40:
        light = "amber"
    else:
        light = "red"
    bg, light_label = TRAFFIC[light]

    st.markdown(
        f'<div style="background:{bg};color:white;border-radius:16px;padding:24px;'
        f'text-align:center;margin-top:16px">'
        f'<div style="font-size:0.9rem;letter-spacing:1px;text-transform:uppercase;opacity:0.9">'
        f'Overall suitability</div>'
        f'<div style="font-size:2.5rem;font-weight:800">{light_label} &middot; {overall_pct}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("All the extra information"):
        st.write(f"Typical mean temperature: **{conditions['temperature_2m_mean_typical']:.1f}°C**")
        st.write(f"Typical daily high: **{conditions['temperature_2m_max_typical']:.1f}°C** "
                 f"&nbsp;&nbsp;low: **{conditions['temperature_2m_min_typical']:.1f}°C**")
        st.write(f"Historically a dry day **{conditions['dry_day_pct']:.0f}%** of the time, "
                 f"this time of year")
        st.caption(f"Based on {int(conditions['n_days'])} historical days within a week of this "
                   f"date, across the full dataset.")

st.divider()
st.caption(
    "Placeholder scoring based on historical climatology, not a live forecast or Adem's model yet. "
    "Swap this out in get_conditions() once the trained model is ready."
)

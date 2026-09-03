"""
Women's Soccer Sport Science Dashboard
----------------------------------------
Three independent tabs:
  1. CMJ Readiness   - daily pre-practice countermovement jump testing
  2. GPS / Catapult   - session load & movement data
  3. Testing Data     - PDF testing reports, filterable by date

The first two tabs each have their own sample-data library (dropdown) AND
their own file uploader, completely independent of the other tab. The
third reads PDF reports straight off disk (see TESTING_PDF_DIR).
"""

import os
import re
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Point Loma Women's Soccer Sport Science Dashboard",
    page_icon="\u26bd",
    layout="wide",
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
TESTING_PDF_DIR = os.path.join(os.path.dirname(__file__), "WSOC_Testing_PDFs")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "pl_soccer_logo.png")

# ---------------------------------------------------------------------------
# Brand colors (extracted directly from the Point Loma Soccer logo)
# ---------------------------------------------------------------------------
PLNU_GREEN = "#085040"
PLNU_GREEN_DARK = "#053a2e"
PLNU_GOLD = "#f8b820"


def force_light_chart(fig):
    """Forces every figure to render with a white/light background and dark
    text regardless of the active Streamlit theme (light, dark, or an
    OS-forced preference) \u2014 charts should never go dark-on-dark or
    light-on-light."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a1a1a"),
        title_font=dict(color="#1a1a1a"),
        legend=dict(font=dict(color="#1a1a1a")),
    )
    fig.update_xaxes(color="#1a1a1a", gridcolor="#e8e6de", linecolor="#cfcdc2")
    fig.update_yaxes(color="#1a1a1a", gridcolor="#e8e6de", linecolor="#cfcdc2")
    return fig


def st_plotly_light(fig, **kwargs):
    """st.plotly_chart wrapper that always forces the light styling above
    and disables Streamlit's automatic theme-sync (which would otherwise
    re-skin the chart dark if the active app theme is dark)."""
    force_light_chart(fig)
    st.plotly_chart(fig, theme=None, width="stretch", **kwargs)

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&display=swap');

        /* ---------------------------------------------------------------
           Typography \u2014 condensed collegiate-athletics style, matching
           the Point Loma Soccer wordmark
        ----------------------------------------------------------------*/
        h1, h2, h3,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        .section-label {{
            font-family: 'Oswald', 'Arial Narrow', sans-serif !important;
            letter-spacing: 0.02em;
        }}
        [data-testid="stMarkdownContainer"] h4 {{
            color: {PLNU_GREEN} !important;
            text-transform: uppercase;
            font-size: 1rem !important;
            font-weight: 700 !important;
            border-bottom: 3px solid {PLNU_GOLD};
            padding-bottom: 6px;
            display: inline-block;
        }}

        /* ---------------------------------------------------------------
           Page background \u2014 clean white, like the athletics site.
           These rules are intentionally broad and unconditional so the
           app stays legible even if a browser/OS forces a dark color
           scheme preference \u2014 light is enforced as the only mode.
        ----------------------------------------------------------------*/
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMain"],
        [data-testid="stSidebar"],
        [data-testid="stBottomBlockContainer"] {{
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }}
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] li {{
            color: #1a1a1a;
        }}

        /* Selectbox / multiselect controls \u2014 force light regardless of
           any dark-mode override, native or Streamlit-internal */
        [data-baseweb="select"] > div,
        [data-baseweb="select"] input,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="menu"] li,
        [data-baseweb="input"] > div {{
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border-color: #cfcdc2 !important;
        }}
        [data-baseweb="menu"] li:hover {{
            background-color: #f0f8f4 !important;
        }}
        .stDateInput input {{
            background-color: #ffffff !important;
            color: #1a1a1a !important;
        }}

        /* ---------------------------------------------------------------
           Header banner \u2014 green bar + tri-stripe accent, like the
           official site's "OFFICIAL SITE OF..." banner
        ----------------------------------------------------------------*/
        .plnu-header-bar {{
            background: linear-gradient(90deg, {PLNU_GREEN_DARK} 0%, {PLNU_GREEN} 100%);
            padding: 20px 26px 16px 26px;
            margin-bottom: 0;
            border-radius: 4px 4px 0 0;
        }}
        .plnu-header-title {{
            font-family: 'Oswald', 'Arial Narrow', sans-serif;
            font-weight: 700;
            font-size: 2.1rem;
            color: #ffffff !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin: 0;
        }}
        .plnu-header-subtitle {{
            color: rgba(255,255,255,0.88) !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }}
        .plnu-stripe {{
            height: 10px;
            background: repeating-linear-gradient(
                90deg,
                {PLNU_GOLD} 0px, {PLNU_GOLD} 100%
            );
            border-top: 2px solid #ffffff;
            border-bottom: 2px solid {PLNU_GREEN_DARK};
            margin-bottom: 22px;
        }}

        /* ---------------------------------------------------------------
           Tabs \u2014 white nav-bar style with gold underline on active
        ----------------------------------------------------------------*/
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 3px solid {PLNU_GOLD};
            background-color: #ffffff;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: #ffffff;
            border-radius: 4px 4px 0 0;
            padding: 10px 22px;
            color: {PLNU_GREEN};
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {PLNU_GREEN} !important;
            color: #ffffff !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {PLNU_GOLD} !important;
        }}

        /* ---------------------------------------------------------------
           KPI metric cards \u2014 white card, green accent, like the
           site's white content cards
        ----------------------------------------------------------------*/
        div[data-testid="stMetric"] {{
            background-color: #ffffff;
            border: 1px solid #e8e6de;
            border-left: 6px solid {PLNU_GREEN};
            border-radius: 6px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }}
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {{
            color: {PLNU_GREEN} !important;
            text-transform: uppercase;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em;
        }}
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {{
            color: #1a1a1a !important;
            font-family: 'Oswald', sans-serif;
        }}
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {{
            color: #555555 !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetricDelta"] svg {{
            display: none;
        }}

        /* ---------------------------------------------------------------
           Filters \u2014 multiselect tags on-brand. Unscoped attribute
           selectors (no ancestor class dependency) since the wrapper
           class name isn't reliable across Streamlit versions.
        ----------------------------------------------------------------*/
        [data-tag] {{
            background-color: {PLNU_GREEN} !important;
            border: 1px solid {PLNU_GOLD} !important;
        }}
        [data-tag] * {{
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }}
        [data-tag] svg {{
            fill: #ffffff !important;
        }}
        .stRadio [role="radiogroup"] label div:first-child {{
            border-color: {PLNU_GREEN} !important;
        }}

        /* ---------------------------------------------------------------
           Chart cards (st.container(border=True)) \u2014 white with a
           green top accent, like the site's article cards
        ----------------------------------------------------------------*/
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: #ffffff;
            border: 1px solid #e8e6de !important;
            border-top: 4px solid {PLNU_GREEN} !important;
            border-radius: 6px !important;
            padding: 6px 10px 2px 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }}

        /* Section labels above chart groups */
        .section-label {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {PLNU_GREEN};
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin: 6px 0 14px 0;
            border-left: 5px solid {PLNU_GOLD};
            padding-left: 10px;
        }}

        /* ---------------------------------------------------------------
           Expander (raw data table) \u2014 green pill header, like the
           site's "Story Archives" button
        ----------------------------------------------------------------*/
        div[data-testid="stExpander"] summary {{
            background-color: {PLNU_GREEN};
            color: #ffffff !important;
            border-radius: 20px;
            padding: 6px 16px;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        }}
        div[data-testid="stExpander"] summary:hover {{
            background-color: {PLNU_GREEN_DARK};
        }}
        div[data-testid="stExpander"] summary * {{
            color: #ffffff !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

CMJ_COLUMNS = [
    "Date", "Match", "Player Name", "CMJ 1", "CMJ 2", "Average",
    "Rolling Baseline", "Readiness Score", "Consecutive Days", "Difference",
    "Z-Score", "% Change", "Rolling SD",
]

# Only these are truly required on upload — Average through Rolling SD are
# derived automatically (see derive_cmj_metrics) when the file doesn't
# already have them.
CMJ_REQUIRED_COLUMNS = ["Date", "Match", "Player Name", "CMJ 1", "CMJ 2"]

GPS_COLUMNS = [
    "Player Name", "Period Name", "Period Number", "Max Acceleration", "Max Deceleration",
    "Acceleration Efforts", "Deceleration Efforts", "Accel + Decel Efforts",
    "Accel + Decel Efforts Per Minute", "Duration", "Distance", "Player Load",
    "Max Velocity", "Max Vel (% Max)", "Meterage Per Minute", "Player Load Per Minute",
    "Work/Rest Ratio", "Max Heart Rate", "Avg Heart Rate", "Max HR (% Max)", "Avg HR (% Max)",
    "HR Exertion", "Red Zone", "Heart Rate Band 1 Duration", "Heart Rate Band 2 Duration",
    "Heart Rate Band 3 Duration", "Heart Rate Band 4 Duration", "Heart Rate Band 5 Duration",
    "Heart Rate Band 6 Duration", "Energy", "High Metabolic Load Distance", "Standing Distance",
    "Walking Distance", "Jogging Distance", "Running Distance", "HI Distance", "Sprint Distance",
    "Sprint Efforts", "Sprint Dist Per Min", "High Speed Distance", "High Speed Efforts",
    "High Speed Distance Per Minute", "Impacts", "Athlete Tags", "Activity Tags", "Game Tags",
    "Athlete Participation Tags", "Period Tags",
]

CMJ_LIBRARY = {
    "Season": "WSOC_CMJ_Season.xlsx",
}

GPS_LIBRARY = {
    "Practice \u2014 Aug 21": "ctr-report-8_21_2026-practice.csv",
    "Match \u2014 Aug 22": "ctr-report_8_22_2026.csv",
    "Practice \u2014 Aug 24": "ctr-report-8_24_2026-practice.csv",
    "Practice \u2014 Aug 25": "ctr-report-8_25_2026-practice.csv",
    "Match \u2014 Aug 27 (vs UCCS)": "ctr-report-8_27_2026-UCCS.csv",
    "Practice \u2014 Aug 28": "ctr-report-8_28_2026-practice.csv",
    "Match \u2014 Aug 29 (vs Metro State)": "ctr-report-8_29_2026-MetroState.csv",
    "Practice \u2014 Aug 31": "ctr-report-8_31_2026-practice.csv",
    "Practice \u2014 Sep 1": "ctr-report-9_1_2026-practice.csv",
    "Practice \u2014 Sep 2": "ctr-report-9_2_2026-practice.csv",
}

# Display label used in GPS chart titles ("Distance - <label>"), matching the
# club's existing daily-report style. Falls back to the uploaded filename.
GPS_SESSION_LABELS = {
    "Practice \u2014 Aug 21": "Friday, August 21 2026",
    "Match \u2014 Aug 22": "Saturday, August 22 2026",
    "Practice \u2014 Aug 24": "Monday, August 24 2026",
    "Practice \u2014 Aug 25": "Tuesday, August 25 2026",
    "Match \u2014 Aug 27 (vs UCCS)": "Thursday, August 27 2026",
    "Practice \u2014 Aug 28": "Friday, August 28 2026",
    "Match \u2014 Aug 29 (vs Metro State)": "Saturday, August 29 2026",
    "Practice \u2014 Aug 31": "Monday, August 31 2026",
    "Practice \u2014 Sep 1": "Tuesday, September 01 2026",
    "Practice \u2014 Sep 2": "Wednesday, September 02 2026",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_excel(path_or_buffer, expected_columns=None):
    name = getattr(path_or_buffer, "name", path_or_buffer)
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(path_or_buffer)
    else:
        df = pd.read_excel(path_or_buffer)
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            st.warning(
                f"Uploaded file is missing expected columns: {', '.join(missing)}. "
                "Charts relying on these fields may not render."
            )
    return df


def data_source_picker(tab_key, library_dict, expected_columns, label):
    """Renders the library-select + uploader UI and returns a loaded dataframe (or None)."""
    st.markdown(f"#### Data source")
    source_mode = st.radio(
        "Choose data source",
        ["Library", "Upload your own file"],
        horizontal=True,
        key=f"{tab_key}_source_mode",
        label_visibility="collapsed",
    )

    df = None
    if source_mode == "Library":
        choice = st.selectbox(
            f"{label} library",
            list(library_dict.keys()),
            key=f"{tab_key}_library_choice",
        )
        path = os.path.join(SAMPLE_DIR, library_dict[choice])
        df = load_excel(path, expected_columns)
        st.caption(f"Loaded from library: **{choice}**")
    else:
        uploaded = st.file_uploader(
            f"Upload {label} Excel/CSV file",
            type=["xlsx", "xls", "csv"],
            key=f"{tab_key}_uploader",
        )
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = load_excel(uploaded, expected_columns)
            st.caption(f"Loaded upload: **{uploaded.name}**")
        else:
            st.info(f"Upload a {label} file, or switch back to Library above.")

    return df


# ---------------------------------------------------------------------------
# Testing-report PDFs (Tab 3)
# ---------------------------------------------------------------------------

# A date embedded in a filename, e.g. the "8-25-2026" in
# "Womens Soccer CMJ_8-25-2026.pdf". Separators may be '-', '_' or '.'
# so a rename to the underscore style used by the GPS exports still parses.
# The lookarounds keep it from matching a fragment of a longer number run.
_DATE_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,4}[-_.]\d{1,2}[-_.]\d{2,4})(?!\d)")


def parse_pdf_date(filename):
    """Extracts the testing date from a PDF filename, or None if absent.

    Accepts the club's M-D-YYYY convention plus ISO (YYYY-M-D) and a
    2-digit year. The format is chosen from where the 4-digit year sits
    rather than by trying formats in order, because strptime's %Y will
    happily read "26" as year 0026 and silently mis-date the file.
    """
    match = _DATE_TOKEN_RE.search(os.path.basename(filename))
    if not match:
        return None
    token = re.sub(r"[_.]", "-", match.group(1))
    parts = token.split("-")
    if len(parts[0]) == 4:
        fmt = "%Y-%m-%d"
    elif len(parts[2]) == 4:
        fmt = "%m-%d-%Y"
    else:
        fmt = "%m-%d-%y"
    try:
        return datetime.strptime(token, fmt).date()
    except ValueError:
        return None


def pdf_display_title(filename):
    """Filename minus its date token and extension, tidied for display."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[_\-\s]+", " ", _DATE_TOKEN_RE.sub("", stem)).strip(" -_")
    return cleaned or stem


def scan_testing_pdfs(directory):
    """Lists testing PDFs as dicts (filename/path/date/title), newest first.

    Deliberately uncached: staff drop new reports into the folder and
    expect a browser refresh to pick them up, so a cache here would just
    hide new files. The listing is a cheap stat of one small directory.
    Undated files sort last but stay visible rather than being dropped.
    """
    if not os.path.isdir(directory):
        return []
    docs = [
        {
            "filename": name,
            "path": os.path.join(directory, name),
            "date": parse_pdf_date(name),
            "title": pdf_display_title(name),
        }
        for name in os.listdir(directory)
        if name.lower().endswith(".pdf")
    ]
    dated = sorted(
        (d for d in docs if d["date"]), key=lambda d: d["date"], reverse=True
    )
    undated = sorted((d for d in docs if not d["date"]), key=lambda d: d["filename"])
    return dated + undated


def pdf_option_label(doc):
    """Selectbox label: '<date> — <title>', or 'Undated — <file>'."""
    when = doc["date"].strftime("%b %d, %Y") if doc["date"] else "Undated"
    return f"{when} — {doc['title']}"


@st.cache_data
def read_pdf_bytes(path, mtime):
    """Raw PDF bytes for the download button. `mtime` is part of the cache
    key so replacing a report in place invalidates the cached copy."""
    with open(path, "rb") as fh:
        return fh.read()


def _stable_seed(name, salt=0):
    """Deterministic per-player seed (Python's built-in hash() is randomized
    per process, so we use md5 to keep this stable across reruns/sessions)."""
    digest = hashlib.md5(f"{name}-{salt}".encode()).hexdigest()
    return int(digest, 16) % (2**32)


def synthetic_rolling_history(player_name, current_value, salt, n_days=28):
    """Builds a plausible 28-day trailing history ending in `current_value`,
    seeded deterministically per player so results are stable across reruns.
    Used to estimate acute:chronic load and rolling 'typical' baselines when
    real multi-session history isn't available (single-day exports/uploads)."""
    rng = np.random.default_rng(_stable_seed(player_name, salt))
    if current_value is None or pd.isna(current_value) or current_value <= 0:
        current_value = 1.0
    baseline = current_value * rng.uniform(0.85, 1.05)
    noise = rng.normal(0, baseline * 0.15, n_days - 1)
    history = np.clip(baseline + noise, baseline * 0.4, baseline * 1.6)
    return np.append(history, current_value)


def acute_chronic_ratio(player_name, current_load, salt=1):
    history = synthetic_rolling_history(player_name, current_load, salt=salt)
    acute = history[-7:].mean()
    chronic = history.mean()
    ratio = acute / chronic if chronic else np.nan
    return acute, chronic, ratio


def synthetic_season_max(player_name, current_value, salt=2):
    rng = np.random.default_rng(_stable_seed(player_name, salt))
    if current_value is None or pd.isna(current_value):
        return current_value
    return round(current_value * rng.uniform(1.02, 1.15), 2)


def typical_baseline(player_name, current_value, salt):
    """Mean of a synthetic rolling history \u2014 used as the 'typical' per-player
    reference value for narrative comparisons (e.g. 'distance was 15% above
    typical')."""
    history = synthetic_rolling_history(player_name, current_value, salt=salt)
    return history.mean()


CMJ_DERIVED_COLUMNS = [
    "Rolling Baseline", "Rolling SD", "Difference", "Z-Score",
    "% Change", "Readiness Score", "Consecutive Days",
]


def derive_cmj_metrics(df, baseline_test_count=30, baseline_window_days=30):
    """Fills in Average / Rolling Baseline / Rolling SD / Difference /
    Z-Score / % Change / Readiness Score / Consecutive Days from raw
    Date + Player Name + CMJ 1 + CMJ 2, replicating the club's original
    CMJ tracking-sheet formulas:
      - Rolling Baseline / Rolling SD: expanding per-player mean / population
        stdev of Average until a player has 30 tests, then a trailing
        30-calendar-day window of Average.
      - Z-Score: (Average - Rolling Baseline) / Rolling SD.
      - Readiness Score: 3 if Z >= 1, 1 if Z <= -1, else 2.
      - Consecutive Days: current streak of Readiness Score == 1 (low)
        days for that player.
    Leaves the file untouched if it already has all of these columns.
    """
    if "Player Name" not in df.columns or "Date" not in df.columns:
        return df
    if all(c in df.columns for c in CMJ_DERIVED_COLUMNS):
        return df

    df = df.copy()
    if "Average" not in df.columns or df["Average"].isna().all():
        if "CMJ 1" in df.columns and "CMJ 2" in df.columns:
            df["Average"] = df[["CMJ 1", "CMJ 2"]].mean(axis=1)
    if "Average" not in df.columns:
        return df

    df = df.sort_values(["Player Name", "Date"], kind="stable").reset_index(drop=True)

    baseline = np.full(len(df), np.nan)
    rolling_sd = np.full(len(df), np.nan)
    for _, idx in df.groupby("Player Name").groups.items():
        idx = list(idx)
        avgs = df.loc[idx, "Average"].to_numpy(dtype=float)
        dates = df.loc[idx, "Date"].to_numpy()
        for pos, row_idx in enumerate(idx):
            if pos + 1 < baseline_test_count:
                window = avgs[: pos + 1]
            else:
                cutoff = dates[pos] - np.timedelta64(baseline_window_days, "D")
                window = avgs[: pos + 1][dates[: pos + 1] >= cutoff]
            window = window[~np.isnan(window)]
            if len(window):
                baseline[row_idx] = window.mean()
                rolling_sd[row_idx] = window.std(ddof=0)

    df["Rolling Baseline"] = baseline
    df["Rolling SD"] = rolling_sd
    df["Difference"] = df["Average"] - df["Rolling Baseline"]
    df["Z-Score"] = np.where(df["Rolling SD"] > 0, df["Difference"] / df["Rolling SD"], np.nan)
    df["% Change"] = np.where(
        df["Rolling Baseline"].fillna(0) != 0, df["Difference"] / df["Rolling Baseline"], 0.0
    )
    df["Readiness Score"] = np.select(
        [df["Z-Score"] >= 1, df["Z-Score"] <= -1], [3, 1], default=2
    ).astype(float)
    df.loc[df["Z-Score"].isna(), "Readiness Score"] = np.nan

    streaks = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("Player Name").groups.items():
        running = 0
        for row_idx in idx:
            running = running + 1 if df.at[row_idx, "Readiness Score"] == 1 else 0
            streaks[row_idx] = running
    df["Consecutive Days"] = streaks

    return df


def flag_readiness(z):
    """Simple 3-tier flag comparing today's jump to the player's rolling
    season average (via Z-Score): Red = notably below, Yellow = below,
    Green = at or above their normal range."""
    if pd.isna(z):
        return "No Data"
    if z <= -1.5:
        return "Red"
    elif z <= -0.5:
        return "Yellow"
    else:
        return "Green"


FLAG_COLORS = {
    "Red": "#d62728",
    "Yellow": "#f0c419",
    "Green": "#2ca02c",
    "No Data": "#b0b0b0",
}


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

header_logo_col, header_text_col = st.columns([1, 4])
with header_logo_col:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width="stretch")
with header_text_col:
    st.markdown(
        """
        <div class="plnu-header-bar">
            <p class="plnu-header-title">Women's Soccer Sport Science Dashboard</p>
            <p class="plnu-header-subtitle">CMJ Readiness &amp; GPS/Catapult Monitoring \u2014 Coaching Staff &amp; Players</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("<div class='plnu-stripe'></div>", unsafe_allow_html=True)

tab_cmj, tab_gps, tab_testing = st.tabs([
    "\U0001F4CA CMJ Readiness",
    "\U0001F6F0\uFE0F GPS / Catapult",
    "\U0001F4C4 Testing Data",
])

# ===========================================================================
# TAB 1: CMJ READINESS
# ===========================================================================
with tab_cmj:
    left, right = st.columns([1, 2.5])

    with left:
        cmj_df = data_source_picker("cmj", CMJ_LIBRARY, CMJ_REQUIRED_COLUMNS, "CMJ Readiness")

    if cmj_df is not None and not cmj_df.empty:
        cmj_df = cmj_df.copy()
        if "Date" in cmj_df.columns:
            cmj_df["Date"] = pd.to_datetime(cmj_df["Date"], errors="coerce")
        cmj_df = derive_cmj_metrics(cmj_df)
        if "Z-Score" in cmj_df.columns:
            cmj_df["Flag"] = cmj_df["Z-Score"].apply(flag_readiness)

        with left:
            st.markdown("#### Filters")
            players = sorted(cmj_df["Player Name"].dropna().unique().tolist()) if "Player Name" in cmj_df.columns else []
            selected_players = st.multiselect(
                "Players", players, default=players[:5] if players else [], key="cmj_players"
            )
            if "Date" in cmj_df.columns and cmj_df["Date"].notna().any():
                min_d, max_d = cmj_df["Date"].min(), cmj_df["Date"].max()
                date_range = st.date_input(
                    "Date range", value=(min_d.date(), max_d.date()),
                    min_value=min_d.date(), max_value=max_d.date(), key="cmj_dates"
                )
            else:
                date_range = None

        filtered = cmj_df.copy()
        if selected_players:
            filtered = filtered[filtered["Player Name"].isin(selected_players)]
        date_ranged = cmj_df.copy()
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]
            date_ranged = date_ranged[(date_ranged["Date"] >= start) & (date_ranged["Date"] <= end)]

        with right:
            # "Today" = the latest date within the selected date range, not
            # necessarily the latest date in the whole file.
            latest_date = date_ranged["Date"].max() if "Date" in date_ranged.columns and not date_ranged.empty else None
            today_df = cmj_df[cmj_df["Date"] == latest_date].copy() if latest_date is not None else cmj_df.iloc[0:0].copy()

            k1, k2, k3 = st.columns(3)
            n_red = (today_df["Flag"] == "Red").sum() if "Flag" in today_df.columns else 0
            n_yellow = (today_df["Flag"] == "Yellow").sum() if "Flag" in today_df.columns else 0
            n_green = (today_df["Flag"] == "Green").sum() if "Flag" in today_df.columns else 0
            k1.metric("\U0001F7E2 Green", int(n_green), delta="Within average range", delta_color="off")
            k2.metric("\U0001F7E1 Yellow", int(n_yellow), delta="Slightly below average range", delta_color="off")
            k3.metric("\U0001F534 Red", int(n_red), delta="Severely below average range", delta_color="off")
            st.caption(f"Testing date: **{latest_date.strftime('%A, %B %d %Y')}**" if pd.notna(latest_date) else "")

        st.markdown("---")

        st.markdown("##### Today's Jump vs. Season Average")
        if not today_df.empty and "Flag" in today_df.columns and "Difference" in today_df.columns:
            plot_df = today_df.copy()
            if selected_players:
                plot_df = plot_df[plot_df["Player Name"].isin(selected_players)]
            plot_df = plot_df.sort_values("Player Name")

            fig = px.bar(
                plot_df, x="Player Name", y="Difference", color="Flag",
                color_discrete_map=FLAG_COLORS,
                category_orders={"Flag": ["Green", "Yellow", "Red", "No Data"]},
                title=f"CMJ Readiness \u2014 {latest_date.strftime('%a, %b %d, %Y')}" if pd.notna(latest_date) else "CMJ Readiness",
                text_auto=".2f",
            )
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.update_layout(
                height=460, xaxis_title="Player", yaxis_title="Difference from Season Average",
                legend_title="Status",
            )
            fig.update_xaxes(tickangle=-40)
            st_plotly_light(fig)
            st.caption(
                "\U0001F7E2 Green = at/above their normal range \u2022 "
                "\U0001F7E1 Yellow = below average, monitor \u2022 "
                "\U0001F534 Red = notably below their season average, flag for follow-up. "
                "Based on each player's Z-Score vs. their own rolling baseline."
            )
        else:
            st.info("No data available for the latest test day with the current filters.")

        with st.expander("View raw data table"):
            st.dataframe(filtered.sort_values("Date", ascending=False), width="stretch")
    else:
        st.info("Choose a library dataset or upload a file on the left to get started.")


# ===========================================================================
# TAB 2: GPS / CATAPULT
# ===========================================================================
with tab_gps:
    left2, right2 = st.columns([1, 2.5])

    with left2:
        gps_source_mode_placeholder = st.session_state.get("gps_source_mode", "Library")
        gps_df = data_source_picker("gps", GPS_LIBRARY, GPS_COLUMNS, "GPS/Catapult")
        gps_library_choice = st.session_state.get("gps_library_choice")

    if gps_df is not None and not gps_df.empty:
        gps_df = gps_df.copy()

        # Session label used in chart titles, matching the club's daily-report style
        if st.session_state.get("gps_source_mode") == "Library" and gps_library_choice in GPS_SESSION_LABELS:
            session_label = GPS_SESSION_LABELS[gps_library_choice]
        else:
            session_label = "Uploaded Session"

        with left2:
            st.markdown("#### Filters")
            players2 = sorted(gps_df["Player Name"].dropna().unique().tolist()) if "Player Name" in gps_df.columns else []
            selected_players2 = st.multiselect(
                "Players", players2, default=players2, key="gps_players"
            )
            periods = sorted(gps_df["Period Name"].dropna().unique().tolist()) if "Period Name" in gps_df.columns else []
            selected_periods = st.multiselect(
                "Period", periods, default=periods, key="gps_periods"
            )

        filtered2 = gps_df.copy()
        if selected_players2:
            filtered2 = filtered2[filtered2["Player Name"].isin(selected_players2)]
        if selected_periods:
            filtered2 = filtered2[filtered2["Period Name"].isin(selected_periods)]

        # ---------------------------------------------------------------
        # Collapse multi-period sessions (e.g. 1st Half / 2nd Half) into
        # one row per player for the team snapshot charts below.
        # ---------------------------------------------------------------
        sum_cols = [c for c in [
            "Distance", "Player Load", "Acceleration Efforts", "Deceleration Efforts",
            "Accel + Decel Efforts", "High Speed Distance", "Sprint Distance",
            "Sprint Efforts", "High Speed Efforts", "Duration", "Impacts",
        ] if c in filtered2.columns]
        max_cols = [c for c in ["Max Velocity", "Max Acceleration"] if c in filtered2.columns]
        first_cols = [c for c in ["Position"] if c in filtered2.columns]

        agg_dict = {c: "sum" for c in sum_cols}
        agg_dict.update({c: "max" for c in max_cols})
        agg_dict.update({c: "first" for c in first_cols})

        if not filtered2.empty and agg_dict:
            team = filtered2.groupby("Player Name", as_index=False).agg(agg_dict)
            if "Distance" in team.columns and "Duration" in team.columns:
                team["Meterage Per Minute"] = (team["Distance"] / team["Duration"]).round(1)
        else:
            team = pd.DataFrame()

        # ---------------------------------------------------------------
        # KPI row + narrative caption, mirroring the club's daily email format
        # ---------------------------------------------------------------
        with right2:
            avg_distance = team["Distance"].mean() if "Distance" in team.columns and not team.empty else None
            avg_pl = team["Player Load"].mean() if "Player Load" in team.columns and not team.empty else None
            avg_hsd = team["High Speed Distance"].mean() if "High Speed Distance" in team.columns and not team.empty else None
            avg_accdec = team["Accel + Decel Efforts"].mean() if "Accel + Decel Efforts" in team.columns and not team.empty else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("D (team avg)", f"{avg_distance/1000:.2f} km" if avg_distance else "\u2014")
            k2.metric("PL (team avg)", f"{avg_pl:.1f}" if avg_pl else "\u2014")
            k3.metric("HSD (team avg)", f"{avg_hsd:.1f} m" if avg_hsd else "\u2014")
            k4.metric("Accel/Decel Efforts (team avg)", f"{avg_accdec:.1f}" if avg_accdec else "\u2014")

            if not team.empty and avg_distance and avg_accdec and avg_hsd:
                typ_distance = team["Player Name"].apply(
                    lambda p: typical_baseline(p, team.loc[team["Player Name"] == p, "Distance"].values[0], salt=10)
                ).mean()
                typ_accdec = team["Player Name"].apply(
                    lambda p: typical_baseline(p, team.loc[team["Player Name"] == p, "Accel + Decel Efforts"].values[0], salt=11)
                ).mean()
                typ_hsd = team["Player Name"].apply(
                    lambda p: typical_baseline(p, team.loc[team["Player Name"] == p, "High Speed Distance"].values[0], salt=12)
                ).mean()

                pct_distance = (avg_distance - typ_distance) / typ_distance * 100 if typ_distance else 0
                pct_accdec = (avg_accdec - typ_accdec) / typ_accdec * 100 if typ_accdec else 0
                pct_hsd = (avg_hsd - typ_hsd) / typ_hsd * 100 if typ_hsd else 0

                st.caption(
                    f"Distance and Accel/Decel Efforts were **{pct_distance:+.0f}%** and **{pct_accdec:+.0f}%** "
                    f"relative to each player's typical rolling levels; HSD was **{pct_hsd:+.0f}%**. "
                    "(Typical levels are estimated from synthetic rolling history \u2014 swap in real "
                    "historical data for production use.)"
                )

        st.markdown("---")

        def threshold_bar_chart(df, metric, title, y_label, color_seq=None):
            fig = px.bar(
                df.sort_values("Player Name"), x="Player Name", y=metric,
                title=f"{title} - {session_label}",
                color_discrete_sequence=color_seq or ["#5B8DEF"],
                text_auto=".2s",
            )
            avg_val = df[metric].mean()
            fig.add_hline(
                y=avg_val, line_dash="dash", line_color=PLNU_GREEN,
                annotation_text=f"Team avg: {avg_val:,.1f}", annotation_position="top left",
            )
            fig.update_layout(
                height=430, xaxis_title="Player Name", yaxis_title=y_label, showlegend=False,
                margin=dict(t=60, b=10, l=10, r=10),
            )
            fig.update_xaxes(tickangle=-40)
            return fig

        def plain_bar_chart(df, metric, title, y_label):
            fig = px.bar(
                df.sort_values("Player Name"), x="Player Name", y=metric,
                title=f"{title} - {session_label}",
                color_discrete_sequence=["#5B8DEF"],
                text_auto=".3s",
            )
            fig.update_layout(
                height=430, xaxis_title="Player Name", yaxis_title=y_label, showlegend=False,
                margin=dict(t=60, b=10, l=10, r=10),
            )
            fig.update_xaxes(tickangle=-40)
            return fig

        def chart_card(fig, caption=None):
            with st.container(border=True):
                st_plotly_light(fig)
                if caption:
                    st.caption(caption)

        def spacer(px_height=26):
            st.markdown(f"<div style='height:{px_height}px'></div>", unsafe_allow_html=True)

        if not team.empty:
            # --- Load & Distance ---
            st.markdown("<div class='section-label'>Load &amp; Distance</div>", unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2, gap="large")
            with r1c1:
                if "Distance" in team.columns:
                    chart_card(threshold_bar_chart(team, "Distance", "Distance", "Distance (m)"))
            with r1c2:
                if "Player Load" in team.columns:
                    chart_card(threshold_bar_chart(team, "Player Load", "Player Load", "Player Load"))
            spacer()

            # --- Load Management ---
            st.markdown("<div class='section-label'>Load Management &amp; Speed</div>", unsafe_allow_html=True)
            r2c1, r2c2 = st.columns(2, gap="large")
            with r2c1:
                if "Player Load" in team.columns:
                    acwr_rows = []
                    for _, row in team.iterrows():
                        acute, chronic, ratio = acute_chronic_ratio(row["Player Name"], row["Player Load"], salt=1)
                        acwr_rows.append({"Player Name": row["Player Name"], "Acute": acute, "Chronic": chronic, "ACWR": ratio})
                    acwr_df = pd.DataFrame(acwr_rows).sort_values("Player Name")

                    fig_acwr = go.Figure()
                    fig_acwr.add_trace(go.Bar(
                        x=acwr_df["Player Name"], y=acwr_df["ACWR"], name="ACWR",
                        marker_color="#636EFA", yaxis="y1",
                        text=acwr_df["ACWR"].round(2), textposition="inside",
                    ))
                    fig_acwr.add_trace(go.Scatter(
                        x=acwr_df["Player Name"], y=acwr_df["Chronic"], name="Chronic Load",
                        mode="lines", line=dict(color=PLNU_GREEN_DARK, shape="hv", width=2), yaxis="y2",
                    ))
                    fig_acwr.add_trace(go.Scatter(
                        x=acwr_df["Player Name"], y=acwr_df["Acute"], name="Acute Load",
                        mode="lines", line=dict(color="#ff4b4b", shape="hv", width=2), yaxis="y2",
                    ))
                    fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor="gray", opacity=0.25, line_width=0, yref="y1")
                    fig_acwr.add_hline(y=0.8, line_dash="dash", line_color="gray", yref="y1")
                    fig_acwr.add_hline(y=1.3, line_dash="dash", line_color="gray", yref="y1")
                    fig_acwr.update_layout(
                        height=430,
                        title=f"Player Load Acute:Chronic Ratio - {session_label}",
                        yaxis=dict(title="Acute:Chronic Ratio"),
                        yaxis2=dict(title="Player Load", overlaying="y", side="right"),
                        xaxis=dict(title="Player Name", tickangle=-40),
                        legend=dict(orientation="h", y=1.16),
                        margin=dict(t=90, b=10, l=10, r=10),
                    )
                    chart_card(fig_acwr, caption="Estimated from synthetic 28-day history \u2014 replace with real historical data when available.")
            with r2c2:
                if "Max Velocity" in team.columns:
                    mv_df = team.sort_values("Player Name").copy()
                    mv_df["Season Max"] = mv_df.apply(
                        lambda r: synthetic_season_max(r["Player Name"], r["Max Velocity"], salt=2), axis=1
                    )
                    color_arg = "Position" if "Position" in mv_df.columns else None
                    fig_mv = px.bar(
                        mv_df, x="Player Name", y="Max Velocity", color=color_arg,
                        title=f"Max Velocity (m/s) - {session_label} \u2014 {len(mv_df)} players",
                        text_auto=".2f",
                    )
                    fig_mv.add_trace(go.Scatter(
                        x=mv_df["Player Name"], y=mv_df["Season Max"], mode="markers+text",
                        text=mv_df["Season Max"].round(1).astype(str), textposition="top center",
                        marker=dict(color=PLNU_GREEN_DARK, size=6), name="Season Max", showlegend=False,
                    ))
                    fig_mv.update_layout(
                        height=430, xaxis_title="Player Name", yaxis_title="Max Velocity",
                        legend=dict(orientation="h", y=1.16),
                        margin=dict(t=90, b=10, l=10, r=10),
                    )
                    fig_mv.update_xaxes(tickangle=-40)
                    chart_card(fig_mv)
            spacer()

            # --- Volume & Efforts ---
            st.markdown("<div class='section-label'>Volume &amp; Efforts</div>", unsafe_allow_html=True)
            r3c1, r3c2 = st.columns(2, gap="large")
            with r3c1:
                if "High Speed Distance" in team.columns:
                    chart_card(threshold_bar_chart(team, "High Speed Distance", "High Speed Distance", "High Speed Distance"))
            with r3c2:
                if "Acceleration Efforts" in team.columns:
                    chart_card(plain_bar_chart(team, "Acceleration Efforts", "Acceleration Efforts", "Acceleration Efforts"))
            spacer(16)
            r4c1, r4c2 = st.columns(2, gap="large")
            with r4c1:
                if "Deceleration Efforts" in team.columns:
                    chart_card(plain_bar_chart(team, "Deceleration Efforts", "Deceleration Efforts", "Deceleration Efforts"))
            with r4c2:
                if "Meterage Per Minute" in team.columns:
                    chart_card(plain_bar_chart(team, "Meterage Per Minute", "Distance/min", "Meterage Per Minute"))

        with st.expander("View raw data table"):
            st.dataframe(filtered2, width="stretch")
    else:
        st.info("Choose a library dataset or upload a file on the left to get started.")


# ===========================================================================
# TAB 3: TESTING DATA (PDF reports)
# ===========================================================================
with tab_testing:
    testing_docs = scan_testing_pdfs(TESTING_PDF_DIR)

    if not testing_docs:
        st.info(
            f"No PDF reports found. Drop testing reports into **{TESTING_PDF_DIR}** "
            "and refresh — include the date in the filename "
            "(e.g. `Womens Soccer CMJ_8-25-2026.pdf`) so they can be filtered by date."
        )
    else:
        left3, right3 = st.columns([1, 2.5])
        dated_docs = [d for d in testing_docs if d["date"]]

        with left3:
            st.markdown("#### Filters")
            testing_range = None
            if dated_docs:
                min_pdf_d = min(d["date"] for d in dated_docs)
                max_pdf_d = max(d["date"] for d in dated_docs)
                testing_range = st.date_input(
                    "Date range",
                    value=(min_pdf_d, max_pdf_d),
                    min_value=min_pdf_d,
                    max_value=max_pdf_d,
                    key="testing_dates",
                )
            else:
                st.caption("No dates found in the report filenames.")

        # While the user is mid-edit st.date_input returns a 1-tuple, so only
        # filter once both ends are set (same guard as the CMJ tab).
        visible_docs = testing_docs
        if testing_range and isinstance(testing_range, tuple) and len(testing_range) == 2:
            start_d, end_d = testing_range
            visible_docs = [
                d for d in testing_docs if d["date"] and start_d <= d["date"] <= end_d
            ]

        with left3:
            if visible_docs:
                selected_label = st.selectbox(
                    "Report",
                    [pdf_option_label(d) for d in visible_docs],
                    key="testing_report",
                )
                selected_doc = next(
                    d for d in visible_docs if pdf_option_label(d) == selected_label
                )
            else:
                selected_doc = None

        with right3:
            k1, k2 = st.columns(2)
            k1.metric("Reports in range", len(visible_docs))
            latest_in_range = max(
                (d["date"] for d in visible_docs if d["date"]), default=None
            )
            k2.metric(
                "Most recent",
                latest_in_range.strftime("%b %d, %Y") if latest_in_range else "—",
            )
            if selected_doc:
                st.caption(f"Viewing: **{selected_doc['filename']}**")

        st.markdown("---")

        if selected_doc is None:
            st.info("No reports fall in the selected date range. Widen the range on the left.")
        else:
            st.markdown(
                f"<div class='section-label'>{selected_doc['title']}"
                + (
                    f" &mdash; {selected_doc['date'].strftime('%A, %B %d %Y')}"
                    if selected_doc["date"]
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.pdf(selected_doc["path"], height=900)
            st.download_button(
                "Download this report",
                data=read_pdf_bytes(
                    selected_doc["path"], os.path.getmtime(selected_doc["path"])
                ),
                file_name=selected_doc["filename"],
                mime="application/pdf",
            )

st.markdown("---")
st.caption(
    "Built for coaching staff and player-facing use. Library datasets are synthetic sample data. "
    "Upload your own CMJ or Catapult export to replace them in either tab \u2014 the two data tabs are "
    "fully independent. Testing Data reads PDF reports from the WSOC_Testing_PDFs folder."
)

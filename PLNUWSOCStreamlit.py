"""
Women's Soccer Sport Science Dashboard
----------------------------------------
Two independent tabs:
  1. CMJ Readiness   - daily pre-practice countermovement jump testing
  2. GPS / Catapult   - session load & movement data

Each tab has its own sample-data library (dropdown) AND its own file
uploader, completely independent of the other tab.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Women's Soccer Sport Science Dashboard",
    page_icon="\u26bd",
    layout="wide",
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

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
    "Preseason \u2014 Week 1-2": "CMJ_Preseason_Week1-2.xlsx",
    "In-Season \u2014 Week 1-2": "CMJ_InSeason_Week1-2.xlsx",
    "This Week": "CMJ_ThisWeek.xlsx",
}

GPS_LIBRARY = {
    "Training \u2014 Aug 18": "GPS_Training_Aug18.xlsx",
    "Match \u2014 vs Rivals FC": "GPS_Match_vs_Rivals.xlsx",
    "Training \u2014 Recovery Day": "GPS_Training_Recovery.xlsx",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_excel(path_or_buffer, expected_columns=None):
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
    if pd.isna(z):
        return "No Data"
    if z <= -1.5:
        return "Low"
    elif z <= -0.5:
        return "Below Average"
    elif z < 0.5:
        return "Normal"
    else:
        return "High"


FLAG_COLORS = {
    "Low": "#d62728",
    "Below Average": "#ff7f0e",
    "Normal": "#2ca02c",
    "High": "#1f77b4",
    "No Data": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("\u26bd Women's Soccer Sport Science Dashboard")
st.caption("CMJ Readiness & GPS/Catapult monitoring \u2014 for coaching staff and players")

tab_cmj, tab_gps = st.tabs(["\U0001F4CA CMJ Readiness", "\U0001F6F0\uFE0F GPS / Catapult"])

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
            today_df = cmj_df[cmj_df["Date"] == latest_date] if latest_date is not None else cmj_df.iloc[0:0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Team Avg Readiness (latest)", f"{today_df['Readiness Score'].mean():.0f}"
                       if "Readiness Score" in today_df.columns and not today_df.empty else "\u2014")
            k2.metric("Team Avg CMJ (cm, latest)", f"{today_df['Average'].mean():.1f}"
                       if "Average" in today_df.columns and not today_df.empty else "\u2014")
            n_low = (today_df["Flag"] == "Low").sum() if "Flag" in today_df.columns else 0
            k3.metric("Players Flagged Low", int(n_low))
            k4.metric("Testing Date", latest_date.strftime("%b %d, %Y") if pd.notna(latest_date) else "\u2014")

        st.markdown("---")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("##### Readiness / CMJ Trend Over Time")
            if not filtered.empty and "Date" in filtered.columns:
                metric_choice = st.radio(
                    "Metric", ["Readiness Score", "Average", "Z-Score"],
                    horizontal=True, key="cmj_metric"
                )
                fig = px.line(
                    filtered.sort_values("Date"), x="Date", y=metric_choice, color="Player Name",
                    markers=True,
                )
                if metric_choice == "Z-Score":
                    fig.add_hline(y=-1.5, line_dash="dash", line_color="red", annotation_text="Low threshold")
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(height=420, legend_title="Player")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Select at least one player to see trends.")

        with c2:
            st.markdown("##### Readiness Flags (latest test day)")
            if "Flag" in today_df.columns and not today_df.empty:
                flag_counts = today_df["Flag"].value_counts().reindex(
                    ["Low", "Below Average", "Normal", "High"]
                ).fillna(0)
                fig2 = go.Figure(go.Bar(
                    x=flag_counts.values, y=flag_counts.index, orientation="h",
                    marker_color=[FLAG_COLORS[f] for f in flag_counts.index],
                ))
                fig2.update_layout(height=420, xaxis_title="Players", yaxis_title="")
                st.plotly_chart(fig2, width="stretch")

        st.markdown("##### Team Readiness Heatmap (Z-Score by Player x Date)")
        if not cmj_df.empty and "Date" in cmj_df.columns:
            pivot = cmj_df.pivot_table(index="Player Name", columns="Date", values="Z-Score", aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            fig3 = px.imshow(
                pivot, color_continuous_scale="RdYlGn", zmin=-2.5, zmax=2.5,
                aspect="auto", labels=dict(color="Z-Score"),
            )
            fig3.update_xaxes(tickformat="%b %d")
            fig3.update_layout(height=max(300, 22 * len(pivot)))
            st.plotly_chart(fig3, width="stretch")

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
        gps_df = data_source_picker("gps", GPS_LIBRARY, GPS_COLUMNS, "GPS/Catapult")

    if gps_df is not None and not gps_df.empty:
        gps_df = gps_df.copy()

        with left2:
            st.markdown("#### Filters")
            players2 = sorted(gps_df["Player Name"].dropna().unique().tolist()) if "Player Name" in gps_df.columns else []
            selected_players2 = st.multiselect(
                "Players", players2, default=players2[:8] if players2 else [], key="gps_players"
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

        with right2:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Distance (m)", f"{filtered2['Distance'].mean():.0f}" if "Distance" in filtered2.columns and not filtered2.empty else "\u2014")
            k2.metric("Avg Player Load", f"{filtered2['Player Load'].mean():.1f}" if "Player Load" in filtered2.columns and not filtered2.empty else "\u2014")
            k3.metric("Avg Sprint Distance (m)", f"{filtered2['Sprint Distance'].mean():.0f}" if "Sprint Distance" in filtered2.columns and not filtered2.empty else "\u2014")
            k4.metric("Avg Max Velocity (m/s)", f"{filtered2['Max Velocity'].mean():.1f}" if "Max Velocity" in filtered2.columns and not filtered2.empty else "\u2014")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Distance & Player Load by Player")
            if not filtered2.empty:
                agg = filtered2.groupby("Player Name", as_index=False).agg(
                    {"Distance": "sum", "Player Load": "sum"}
                ).sort_values("Distance", ascending=True)
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(y=agg["Player Name"], x=agg["Distance"], name="Distance (m)", orientation="h"))
                fig4.update_layout(height=max(350, 24 * len(agg)), xaxis_title="Distance (m)", barmode="group")
                st.plotly_chart(fig4, width="stretch")

        with c2:
            st.markdown("##### Sprint & High-Speed Distance by Player")
            if not filtered2.empty and "Sprint Distance" in filtered2.columns:
                agg2 = filtered2.groupby("Player Name", as_index=False).agg(
                    {"Sprint Distance": "sum", "High Speed Distance": "sum"}
                ).sort_values("Sprint Distance", ascending=True)
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(y=agg2["Player Name"], x=agg2["Sprint Distance"], name="Sprint Distance (m)", orientation="h"))
                fig5.add_trace(go.Bar(y=agg2["Player Name"], x=agg2["High Speed Distance"], name="High Speed Distance (m)", orientation="h"))
                fig5.update_layout(height=max(350, 24 * len(agg2)), barmode="group", xaxis_title="Meters")
                st.plotly_chart(fig5, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### Max Velocity vs. Sprint Distance")
            if not filtered2.empty and "Max Velocity" in filtered2.columns:
                fig6 = px.scatter(
                    filtered2, x="Max Velocity", y="Sprint Distance", color="Player Name",
                    size="Player Load" if "Player Load" in filtered2.columns else None,
                    hover_data=["Period Name"] if "Period Name" in filtered2.columns else None,
                )
                fig6.update_layout(height=380)
                st.plotly_chart(fig6, width="stretch")

        with c4:
            st.markdown("##### Acceleration / Deceleration Efforts")
            if not filtered2.empty and "Acceleration Efforts" in filtered2.columns:
                agg3 = filtered2.groupby("Player Name", as_index=False).agg(
                    {"Acceleration Efforts": "sum", "Deceleration Efforts": "sum"}
                ).sort_values("Acceleration Efforts", ascending=True)
                fig7 = go.Figure()
                fig7.add_trace(go.Bar(y=agg3["Player Name"], x=agg3["Acceleration Efforts"], name="Accel Efforts", orientation="h"))
                fig7.add_trace(go.Bar(y=agg3["Player Name"], x=agg3["Deceleration Efforts"], name="Decel Efforts", orientation="h"))
                fig7.update_layout(height=380, barmode="group", xaxis_title="Efforts")
                st.plotly_chart(fig7, width="stretch")

        st.markdown("##### Heart Rate Band Distribution (selected players)")
        hr_band_cols = [f"Heart Rate Band {i} Duration" for i in range(1, 7)]
        if not filtered2.empty and all(c in filtered2.columns for c in hr_band_cols):
            hr_agg = filtered2.groupby("Player Name", as_index=False)[hr_band_cols].sum()
            hr_melt = hr_agg.melt(id_vars="Player Name", var_name="Band", value_name="Seconds")
            fig8 = px.bar(
                hr_melt, x="Player Name", y="Seconds", color="Band",
                labels={"Seconds": "Duration (sec)"},
            )
            fig8.update_layout(height=400, xaxis_tickangle=-40)
            st.plotly_chart(fig8, width="stretch")

        with st.expander("View raw data table"):
            st.dataframe(filtered2, width="stretch")
    else:
        st.info("Choose a library dataset or upload a file on the left to get started.")

st.markdown("---")
st.caption(
    "Built for coaching staff and player-facing use. Library datasets are synthetic sample data. "
    "Upload your own CMJ or Catapult export to replace them in either tab \u2014 the two tabs are fully independent."
)

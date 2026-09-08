"""Fatigue watchlist: joins the CMJ readiness sheet to the GPS/Catapult
exports and surfaces players carrying both signals at once.

A player lands on the watchlist only when BOTH halves agree:
  - CMJ:  their most recent jump sits in the bottom quartile of their own
          season. Fatigue shows up in a countermovement jump as a *lower*
          number, so this is the mirror of the GPS rule below.
  - GPS:  their trailing 7-day load is above their own historical 75th
          percentile on either load metric.

Every threshold is per-player. A 5'2" outside back and a centre back who
covers 11km a game have no business being measured against a squad average,
and the whole point of the tab is to catch a player who is heavy *for her*.

Names are resolved through `roster` before joining -- see that module for why
the two files disagree about what people are called.
"""

import numpy as np
import pandas as pd

import roster

# Every rostered player, so the board can show one row each even for a player
# who produced no usable reading on either side this season.
ROSTER_ORDER = list(roster.ROSTER)

# GPS metrics standing in for "fatigue-inducing work". Player Load is the
# accumulated-load headline number; the high-speed pair captures the short,
# explosive session that Player Load alone under-reports. A player trips the
# GPS half if EITHER is above her own 75th percentile, so a sprint-heavy
# Tuesday counts as much as a long Saturday.
LOAD_METRICS = ["Player Load", "HI + Sprint Distance"]

WINDOW_DAYS = 7
PERCENTILE = 0.75

# A player needs a few prior windows before "her own 75th percentile" means
# anything. Below this we report insufficient history rather than flagging or
# clearing her -- an empty history should never read as "clear".
MIN_HISTORY = 4

# Captures known to be invalid, dropped before anything is scored.
#
# A dead pod does not read as missing data, it reads as a very easy week, and
# that is worse than a gap in two ways: the player is reported "Clear" on
# numbers that describe nothing, and the near-zero windows sit in her own
# reference distribution and drag her personal 75th percentile down. Lila
# Jones' threshold fell to about a third of the squad's that way, so a normal
# week would have read as 37% above her own p75 once the pod was swapped --
# a false flag lasting until enough good weeks diluted the zeros.
#
# `end` is None while a fault is open. Dates are inclusive.
EXCLUDED_CAPTURES = [
    {
        "player": "Lila Jones",
        "start": "2026-09-01",
        "end": None,
        "reason": "faulty pod awaiting replacement",
    },
]


def _excluded_mask(df):
    """Rows covered by an EXCLUDED_CAPTURES entry, matched on canonical name so
    an exclusion holds however the export spelled the player."""
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    for rule in EXCLUDED_CAPTURES:
        player = roster.resolve(rule["player"]) or rule["player"]
        covered = df["Player Name"] == player
        start = pd.to_datetime(rule.get("start")) if rule.get("start") else None
        end = pd.to_datetime(rule.get("end")) if rule.get("end") else None
        if start is not None:
            covered &= df["Date"] >= start
        if end is not None:
            covered &= df["Date"] <= end
        mask |= covered
    return mask


def active_exclusion(player, as_of=None):
    """The reason a player's GPS is being ignored as of `as_of`, or None.

    Used to label her on the board rather than leaving her to be scored on
    whatever partial history survives the exclusion -- a stale window is no
    more meaningful than the bad one it replaced.
    """
    canonical = roster.resolve(player) or player
    as_of = pd.to_datetime(as_of) if as_of is not None else None
    for rule in EXCLUDED_CAPTURES:
        if (roster.resolve(rule["player"]) or rule["player"]) != canonical:
            continue
        start = pd.to_datetime(rule.get("start")) if rule.get("start") else None
        end = pd.to_datetime(rule.get("end")) if rule.get("end") else None
        if as_of is not None:
            if start is not None and as_of < start:
                continue
            if end is not None and as_of > end:
                continue
        return rule.get("reason", "excluded capture")
    return None


def prepare_gps(frames):
    """Stacks dated GPS sessions into one name-normalised frame, one row per
    player per day.

    `frames` is an iterable of (date, is_match, dataframe). The GPS tab loads a
    single session at a time; a rolling load window needs the season at once.
    """
    prepared = []
    for date, is_match, session in frames:
        if session is None or session.empty or "Player Name" not in session.columns:
            continue
        session = session.copy()
        session["Date"] = pd.to_datetime(date)
        session["Is Match"] = bool(is_match)
        prepared.append(session)

    if not prepared:
        return pd.DataFrame()

    df = pd.concat(prepared, ignore_index=True)
    df["Raw Name"] = df["Player Name"].astype(str).str.strip()
    df["Player Name"] = roster.canonicalize(df["Raw Name"])

    for col in ("Player Load", "HI Distance", "Sprint Distance", "Distance"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["HI + Sprint Distance"] = df["HI Distance"].fillna(0) + df["Sprint Distance"].fillna(0)

    # Catapult exports a row per period. Today's files are one row per player,
    # but sum rather than assume that holds for a future multi-period export.
    agg = {metric: "sum" for metric in LOAD_METRICS}
    agg["Distance"] = "sum"
    agg["Is Match"] = "max"
    # Drop known-bad captures before any aggregation, so they reach neither a
    # player's current window nor the distribution it is compared against.
    df = df[~_excluded_mask(df)]

    daily = df.groupby(["Player Name", "Date"], as_index=False).agg(agg)
    return daily.sort_values(["Player Name", "Date"], kind="stable").reset_index(drop=True)


def rolling_load(daily, window_days=WINDOW_DAYS):
    """Per player, the trailing `window_days` *calendar*-day sum of each load
    metric as of every session date. Calendar-based rather than session-based so
    a light week and a congested week are not treated as equivalent."""
    if daily.empty:
        return daily

    out = daily.copy()
    for metric in LOAD_METRICS:
        out[f"{metric} ({window_days}d)"] = np.nan
    out["Window Complete"] = False

    for _, idx in out.groupby("Player Name").groups.items():
        idx = list(idx)
        dates = out.loc[idx, "Date"].to_numpy()
        first_seen = dates[0]
        for metric in LOAD_METRICS:
            values = out.loc[idx, metric].to_numpy(dtype=float)
            col = f"{metric} ({window_days}d)"
            for pos, row_idx in enumerate(idx):
                cutoff = dates[pos] - np.timedelta64(window_days - 1, "D")
                window = values[: pos + 1][dates[: pos + 1] >= cutoff]
                window = window[~np.isnan(window)]
                out.at[row_idx, col] = window.sum() if len(window) else np.nan

        # A window that starts before the player's first session is only
        # partially filled, so its sum is small for a structural reason rather
        # than a training one. Early-season windows like that would otherwise
        # sit at the bottom of the reference distribution and drag the 75th
        # percentile down far enough to flag nearly the whole squad.
        for pos, row_idx in enumerate(idx):
            span = (dates[pos] - first_seen) / np.timedelta64(1, "D")
            out.at[row_idx, "Window Complete"] = span >= window_days - 1
    return out


def cmj_state(cmj_df, percentile=PERCENTILE):
    """Per player: is her most recent CMJ in the bottom quartile of her own
    season? Uses the whole season as the reference distribution, including the
    latest test -- with a short season, excluding it would leave too little to
    take a quantile over."""
    if cmj_df is None or cmj_df.empty:
        return pd.DataFrame()
    if "Average" not in cmj_df.columns or "Date" not in cmj_df.columns:
        return pd.DataFrame()

    df = cmj_df.dropna(subset=["Average", "Date"]).copy()
    df["Player Name"] = roster.canonicalize(df["Player Name"].astype(str).str.strip())

    rows = []
    for player, group in df.groupby("Player Name"):
        group = group.sort_values("Date", kind="stable")
        latest = group.iloc[-1]
        history = group["Average"].to_numpy(dtype=float)
        history = history[~np.isnan(history)]
        threshold = np.quantile(history, 1 - percentile) if len(history) else np.nan
        z = latest.get("Z-Score", np.nan)
        rows.append({
            "Player Name": player,
            "CMJ Latest": float(latest["Average"]),
            "CMJ Date": latest["Date"],
            "CMJ Threshold": threshold,
            "CMJ Baseline": float(np.median(history)) if len(history) else np.nan,
            "CMJ Z-Score": float(z) if pd.notna(z) else np.nan,
            "CMJ Tests": int(len(history)),
            "CMJ Fatigued": bool(
                len(history) >= MIN_HISTORY and latest["Average"] <= threshold
            ),
        })
    return pd.DataFrame(rows)


def gps_state(daily, window_days=WINDOW_DAYS, percentile=PERCENTILE):
    """Per player: is her current rolling load above her own historical 75th
    percentile on either metric? The percentile is taken over that player's
    PRIOR windows only, so today's spike cannot inflate the bar it is being
    measured against."""
    if daily.empty:
        return pd.DataFrame()

    rolled = rolling_load(daily, window_days)
    rows = []
    for player, group in rolled.groupby("Player Name"):
        group = group.sort_values("Date", kind="stable")
        latest = group.iloc[-1]
        record = {
            "Player Name": player,
            "GPS Date": latest["Date"],
            "GPS Sessions": int(len(group)),
        }
        # Only fully-populated windows form the reference distribution.
        complete = group["Window Complete"].to_numpy(dtype=bool)
        triggers, prior_counts = [], []
        for metric in LOAD_METRICS:
            series = group[f"{metric} ({window_days}d)"].to_numpy(dtype=float)
            current = series[-1]
            prior = series[:-1][complete[:-1]]
            prior = prior[~np.isnan(prior)]
            threshold = np.quantile(prior, percentile) if len(prior) else np.nan
            record[f"{metric} Current"] = current
            record[f"{metric} Threshold"] = threshold
            prior_counts.append(len(prior))
            if (
                len(prior) >= MIN_HISTORY - 1
                and pd.notna(current)
                and pd.notna(threshold)
                and current > threshold
            ):
                triggers.append(metric)
        record["GPS Windows"] = max(prior_counts) + 1 if prior_counts else 0
        record["GPS Triggers"] = triggers
        record["GPS Fatigued"] = bool(triggers)
        rows.append(record)
    return pd.DataFrame(rows)


def build_board(cmj_df, daily_gps):
    """Joins the two halves on canonical roster names.

    Returns every player either source knows about, with a `Status` explaining
    where each stands. Players absent from one source are kept and labelled
    rather than dropped -- silently vanishing from a fatigue tool is the
    failure mode worth engineering against.
    """
    cmj = cmj_state(cmj_df)
    gps = gps_state(daily_gps)
    if cmj.empty and gps.empty:
        return pd.DataFrame()

    if cmj.empty or gps.empty:
        board = (gps if cmj.empty else cmj).copy()
        board["CMJ Fatigued"] = board.get("CMJ Fatigued", False)
        board["GPS Fatigued"] = board.get("GPS Fatigued", False)
        board["On Watchlist"] = False
        board["Status"] = "No CMJ data" if cmj.empty else "No GPS data"
        board["Photo"] = board["Player Name"].map(roster.image_path)
        return board.reset_index(drop=True)

    board = cmj.merge(gps, on="Player Name", how="outer")
    board["CMJ Fatigued"] = board["CMJ Fatigued"].fillna(False).astype(bool)
    board["GPS Fatigued"] = board["GPS Fatigued"].fillna(False).astype(bool)
    board["On Watchlist"] = board["CMJ Fatigued"] & board["GPS Fatigued"]

    # Latest session in the data, used to ask whether an exclusion is still open.
    as_of = board["GPS Date"].max() if board["GPS Date"].notna().any() else None

    def status(row):
        if pd.isna(row.get("CMJ Latest")):
            return "No CMJ data"
        # An open exclusion outranks whatever survives it: scoring her on a
        # stale pre-fault window would clear or flag her on the wrong week.
        if active_exclusion(row["Player Name"], as_of):
            return "GPS excluded"
        if pd.isna(row.get("GPS Date")):
            return "No GPS data"
        if row.get("CMJ Tests", 0) < MIN_HISTORY or row.get("GPS Windows", 0) < MIN_HISTORY:
            return "Not enough history"
        if row["On Watchlist"]:
            return "Watchlist"
        if row["CMJ Fatigued"]:
            return "CMJ only"
        if row["GPS Fatigued"]:
            return "Load only"
        return "Clear"

    board["Status"] = board.apply(status, axis=1)
    # "Watchlist" is what the tab leads with, so a row whose status was
    # downgraded for thin history must not still read as flagged.
    board.loc[board["Status"] != "Watchlist", "On Watchlist"] = False

    # Blank the GPS columns for an excluded player. What survives an exclusion
    # is her last pre-fault window, and printed beside a threshold built from
    # the same short history it reads as a large overload -- the stale numbers
    # mislead exactly as much as the status they sit next to would have.
    excluded = board["Status"] == "GPS excluded"
    if excluded.any():
        gps_columns = [c for c in board.columns
                       if c.startswith(tuple(LOAD_METRICS)) or c.startswith("GPS ")]
        for column in gps_columns:
            if column == "GPS Triggers":
                board.loc[excluded, column] = board.loc[excluded, column].apply(lambda _: [])
            else:
                board.loc[excluded, column] = np.nan
        board.loc[excluded, "GPS Fatigued"] = False
    # A rostered player with no usable reading on either side matches nothing to
    # merge and would otherwise leave no row at all. Emma Blakely is on the CMJ
    # sheet every test day with the jump columns blank and has never worn a pod,
    # so she vanished from the board entirely -- the one outcome this tab must
    # not produce, since an absent player looks identical to a fine one.
    for player in ROSTER_ORDER:
        if player not in set(board["Player Name"]):
            board = pd.concat(
                [board, pd.DataFrame([{
                    "Player Name": player,
                    "Status": "No data",
                    "CMJ Fatigued": False,
                    "GPS Fatigued": False,
                    "On Watchlist": False,
                }])],
                ignore_index=True,
            )

    board["Photo"] = board["Player Name"].map(roster.image_path)

    return board.sort_values(
        ["On Watchlist", "CMJ Z-Score"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def squad_load_trend(daily, window_days=WINDOW_DAYS, metric="Player Load"):
    """Squad-median rolling load per session date, over complete windows only.

    While the squad is ramping, almost everyone sits near a season high by
    construction and the load half of the test flags widely -- true early in
    preseason, misleading once volume plateaus. Reporting the trend from the
    data keeps the tab's caveat honest instead of freezing whatever was true
    the day it was written.
    """
    if daily.empty:
        return pd.Series(dtype=float)
    rolled = rolling_load(daily, window_days)
    rolled = rolled[rolled["Window Complete"]]
    if rolled.empty:
        return pd.Series(dtype=float)
    return rolled.groupby("Date")[f"{metric} ({window_days}d)"].median().sort_index()


def load_phase(daily, window_days=WINDOW_DAYS, lookback=3, threshold=0.05,
               peak_ratio=0.98, deload_ratio=0.85):
    """Where squad load sits relative to its own season: "ramping", "steady",
    "deload", or None when history is too short to say.

    Each phase changes how an empty watchlist should be read, so the tab says
    which one it is rather than leaving a coach to assume:

      ramping  -- the latest window is both up on recent sessions and still at
                  the season peak. Nearly everyone sits near a personal high by
                  construction, so the load half flags widely.
      deload   -- the latest window is well below the peak. Almost nobody can
                  clear their own 75th percentile, so few or no players trip
                  the load half and an empty watchlist reflects the taper
                  rather than a recovered squad.
      steady   -- in between, where both halves discriminate normally.

    The peak comparison is what keeps this honest across a turning point:
    checking only against N sessions ago straddles the peak itself, so a squad
    two sessions into a taper would still read as ramping.
    """
    trend = squad_load_trend(daily, window_days)
    if len(trend) < lookback + 1:
        return None

    latest, peak = trend.iloc[-1], trend.max()
    if latest < deload_ratio * peak:
        return "deload"
    rising = latest / trend.iloc[-(lookback + 1)] - 1 > threshold
    if rising and latest >= peak_ratio * peak:
        return "ramping"
    return "steady"


def load_pct(row, metric):
    """How far above (or below) her own threshold a player's current window
    sits, as a percentage. Used for the '+18% vs. her usual' style caption."""
    current = row.get(f"{metric} Current")
    threshold = row.get(f"{metric} Threshold")
    if pd.isna(current) or pd.isna(threshold) or not threshold:
        return None
    return (current / threshold - 1.0) * 100.0

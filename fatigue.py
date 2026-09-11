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

# A drop has to clear the protocol's own measurement noise before it means
# anything. The quartile rule alone flags the bottom 25% of a player's own
# season by construction: someone is always in it, whether or not she changed.
# On the season to Sep 8 it fired on eleven players, eight of whom had dropped
# less than the jump test can actually resolve.
#
# Two trials are recorded per test, so the noise is measurable rather than
# assumed -- see cmj_detectable_change(). This constant is only the fallback
# for a sheet that arrives with the trial columns already averaged away.
# Measured over the 242 paired trials to Sep 8 it comes out at 0.65 cm.
CMJ_NOISE_FALLBACK = 0.65

# One-sided, because the only question ever asked of a jump here is "is she
# LOWER than her baseline", never "is she different from it". A two-sided
# 1.96 prices the wrong question and costs about 0.12 cm of sensitivity for
# nothing -- on the season to Sep 8 that was the difference between catching
# Riley Johnson's 0.65 cm drop and missing it.
CMJ_NOISE_Z = 1.645

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
# `player` may name one athlete or several -- a pod fault is usually personal,
# but travel contamination hits everyone whose pod shared the vehicle. `end` is
# None while a fault is open. Dates are inclusive.
EXCLUDED_CAPTURES = [
    {
        "player": "Lila Jones",
        "start": "2026-09-01",
        "end": None,
        "reason": "faulty pod awaiting replacement",
    },
    {
        # Five pods ran through the coach trip to Cal Poly Pomona: 4.6 hours,
        # 27 km and a top speed of 122 km/h each, against 21-24 km/h for the
        # team-mates whose pods were switched off. Their real match work is
        # inside those totals and cannot be separated from the drive, so the
        # honest reading is that Sep 5 is unknown for these five rather than
        # enormous. Left in, it would have raised their load thresholds far
        # enough to mask a genuinely hard week later in the season.
        "player": [
            "Madilyn Audet",
            "Riley Johnson",
            "Grace Nelson",
            "Gianna Masinter",
            "Abby Wright",
        ],
        "start": "2026-09-05",
        "end": "2026-09-05",
        "reason": "pod left running during travel to CPP",
    },
]


# Captures that are real but stop short of the session they belong to.
# Distinct from EXCLUDED_CAPTURES: an exclusion says the numbers are junk, a
# partial capture says they are a true lower bound. That makes them useful in
# one direction only -- a window already over a player's threshold on what was
# recorded is over it for certain, so it may still flag her; a window under it
# says nothing, so it may not clear her. Partial windows are also kept out of
# every reference distribution, since a threshold built from understated weeks
# would lower the bar for the rest of the season.
#
# Dropping the day instead would be worse, not safer: the window would then
# lose the recorded minutes as well as the missing ones.
#
# Same shape as EXCLUDED_CAPTURES, except `player` may also be None for a
# capture that stopped for the whole squad at once. Remove an entry once the
# full export replaces the partial one, as the Sep 10 LA match's did.
PARTIAL_CAPTURES = []


# Context that changes how a row should be READ without changing how it is
# scored. Deliberately separate from EXCLUDED_CAPTURES: an exclusion says the
# numbers are junk, a note says the numbers are real and mean something other
# than the obvious.
#
# A return-to-play ramp trips the GPS half almost by construction. The rolling
# window is meant to climb week on week, so it sits above the player's own
# prior 75th percentile for as long as the progression is working -- Kylee
# Jerome's has risen every window since her first session. Excluding her would
# be wrong twice over: her pod recorded genuine running, and dropping the ramp
# would leave her scored on nothing. But an unexplained flag is worse than no
# flag, because a coach who learns to discount it will also discount the first
# real one after she comes off the ramp.
#
# Same shape as EXCLUDED_CAPTURES: `player` may name one athlete or several,
# `end` is None while the note still applies, dates inclusive.
PLAYER_NOTES = [
    {
        "player": "Kylee Jerome",
        # When the RTP block began. Her first pod capture came three days
        # later, on Aug 27, so every GPS window she has is inside the ramp.
        "start": "2026-08-24",
        "end": None,
        "note": "Returning to play \u2014 rising load is a planned ramp",
    },
]


# A top speed no footballer reaches. The squad's own season high is about
# 7.3 m/s and a world-class sprinter peaks near 12, so anything past this did
# not come from someone running. The Sep 5 travel rows read 34 m/s.
IMPLAUSIBLE_VELOCITY = 12.0


def suspect_captures(daily, limit=IMPLAUSIBLE_VELOCITY):
    """Player-days whose peak speed is too high to have come from running.

    A pod left switched on in a vehicle produces a plausible-looking load total
    attached to an impossible top speed, and it inflates the player's own
    threshold rather than obviously breaking anything. This was caught by hand
    once; the check exists so the next one is caught by the tab. It reports
    rather than drops -- deciding a capture is junk is a judgement for whoever
    knows what happened that day, recorded in EXCLUDED_CAPTURES.
    """
    if daily.empty or "Max Velocity" not in daily.columns:
        return pd.DataFrame(columns=["Player Name", "Date", "Max Velocity"])
    suspect = daily[daily["Max Velocity"] > limit]
    return suspect[["Player Name", "Date", "Max Velocity"]].sort_values(
        ["Date", "Player Name"]
    ).reset_index(drop=True)


def _rule_players(rule):
    """Canonical names an exclusion rule covers. `player` may be one name or a
    list, resolved through the roster so a rule holds however an export spelled
    the athlete. None means the rule covers everyone."""
    named = rule.get("player")
    if named is None:
        return None
    if isinstance(named, str):
        named = [named]
    return [roster.resolve(name) or name for name in named]


def _rule_mask(df, rules):
    """Rows covered by any entry in `rules`, matched on canonical name so a
    rule holds however the export spelled the player."""
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    for rule in rules:
        players = _rule_players(rule)
        covered = (df["Player Name"].isin(players) if players is not None
                   else pd.Series(True, index=df.index))
        start = pd.to_datetime(rule.get("start")) if rule.get("start") else None
        end = pd.to_datetime(rule.get("end")) if rule.get("end") else None
        if start is not None:
            covered &= df["Date"] >= start
        if end is not None:
            covered &= df["Date"] <= end
        mask |= covered
    return mask


def _active_rule(rules, player, as_of=None):
    """First rule in `rules` covering `player` at `as_of`, or None.

    EXCLUDED_CAPTURES and PLAYER_NOTES are matched identically and differ only
    in which field the caller reads off the match, so the date-and-name walk
    lives here once.
    """
    canonical = roster.resolve(player) or player
    as_of = pd.to_datetime(as_of) if as_of is not None else None
    for rule in rules:
        players = _rule_players(rule)
        if players is not None and canonical not in players:
            continue
        start = pd.to_datetime(rule.get("start")) if rule.get("start") else None
        end = pd.to_datetime(rule.get("end")) if rule.get("end") else None
        if as_of is not None:
            if start is not None and as_of < start:
                continue
            if end is not None and as_of > end:
                continue
        return rule
    return None


def active_exclusion(player, as_of=None):
    """The reason a player's GPS is being ignored as of `as_of`, or None.

    Used to label her on the board rather than leaving her to be scored on
    whatever partial history survives the exclusion -- a stale window is no
    more meaningful than the bad one it replaced.
    """
    rule = _active_rule(EXCLUDED_CAPTURES, player, as_of)
    return rule.get("reason", "excluded capture") if rule else None


def active_note(player, as_of=None):
    """The note that should ride along with a player's row as of `as_of`, or
    None. Annotation only -- nothing here reaches the scoring, so a noted
    player is flagged or cleared on exactly the numbers she would have been
    without one."""
    rule = _active_rule(PLAYER_NOTES, player, as_of)
    return rule.get("note") if rule else None


def partial_captures_between(start, end):
    """PARTIAL_CAPTURES entries whose dates overlap `start`..`end` inclusive,
    for naming the cause wherever a partial capture changes what a reader sees.
    """
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    found = []
    for rule in PARTIAL_CAPTURES:
        rule_start = pd.to_datetime(rule.get("start")) if rule.get("start") else None
        rule_end = pd.to_datetime(rule.get("end")) if rule.get("end") else None
        if rule_start is not None and rule_start > end:
            continue
        if rule_end is not None and rule_end < start:
            continue
        found.append(rule)
    return found


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

    for col in ("Player Load", "HI Distance", "Sprint Distance", "Distance",
                "Max Velocity"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["HI + Sprint Distance"] = df["HI Distance"].fillna(0) + df["Sprint Distance"].fillna(0)

    # Catapult exports a row per period. Today's files are one row per player,
    # but sum rather than assume that holds for a future multi-period export.
    agg = {metric: "sum" for metric in LOAD_METRICS}
    agg["Distance"] = "sum"
    agg["Is Match"] = "max"
    # Carried through so a capture can be sanity-checked after the fact; a peak
    # speed is a property of the day, not something to add up.
    agg["Max Velocity"] = "max"
    # A day is a lower bound if any part of it is.
    agg["Partial"] = "max"
    # Drop known-bad captures before any aggregation, so they reach neither a
    # player's current window nor the distribution it is compared against.
    df = df[~_rule_mask(df, EXCLUDED_CAPTURES)]
    df = df.assign(Partial=_rule_mask(df, PARTIAL_CAPTURES))

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
    out["Window Partial"] = False
    partial_days = (out["Partial"].eq(True) if "Partial" in out.columns
                    else pd.Series(False, index=out.index))

    for _, idx in out.groupby("Player Name").groups.items():
        idx = list(idx)
        dates = out.loc[idx, "Date"].to_numpy()
        partial = partial_days.loc[idx].to_numpy(dtype=bool)
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
            # A window holding any partial day is a lower bound on the week
            # (see PARTIAL_CAPTURES).
            cutoff = dates[pos] - np.timedelta64(window_days - 1, "D")
            in_window = dates[: pos + 1] >= cutoff
            out.at[row_idx, "Window Partial"] = bool(partial[: pos + 1][in_window].any())
    return out


def cmj_detectable_change(cmj_df, fallback=CMJ_NOISE_FALLBACK):
    """Smallest jump change distinguishable from measurement noise, in cm.

    Derived from the squad's own repeat trials rather than a literature value,
    so it tracks this protocol on this equipment: the spread of the two trials
    within a test gives the typical error of one trial, the average of two is
    sqrt(2) tighter than that, and CMJ_NOISE_Z * sqrt(2) * TE is the drop that
    clears it at 95% one-sided. Falls back to the documented constant when a
    sheet has no trial columns to measure.
    """
    if cmj_df is None or "CMJ 1" not in cmj_df.columns or "CMJ 2" not in cmj_df.columns:
        return fallback
    pairs = cmj_df[["CMJ 1", "CMJ 2"]].dropna()
    if len(pairs) < 30:
        # Too few repeats to estimate a spread worth trusting.
        return fallback
    typical_error = (pairs["CMJ 1"] - pairs["CMJ 2"]).std() / np.sqrt(2)
    return float(CMJ_NOISE_Z * np.sqrt(2) * (typical_error / np.sqrt(2)))


def cmj_state(cmj_df, percentile=PERCENTILE):
    """Per player: is her most recent CMJ in the bottom quartile of her own
    season AND down by more than the test can resolve? Uses the whole season as
    the reference distribution, including the latest test -- with a short
    season, excluding it would leave too little to take a quantile over.

    Both conditions are needed. The quartile alone is relative, so it always
    names somebody; the noise floor alone is absolute, so it would fire on a
    player whose jump is merely low today. Together they mean "low for her, by
    an amount the equipment can actually see" (see cmj_detectable_change)."""
    if cmj_df is None or cmj_df.empty:
        return pd.DataFrame()
    if "Average" not in cmj_df.columns or "Date" not in cmj_df.columns:
        return pd.DataFrame()

    df = cmj_df.dropna(subset=["Average", "Date"]).copy()
    df["Player Name"] = roster.canonicalize(df["Player Name"].astype(str).str.strip())

    # Measured across the whole sheet, not per player: one athlete's handful of
    # repeats is far too few to estimate her own noise, and the error belongs to
    # the protocol rather than to her.
    detectable = cmj_detectable_change(cmj_df)

    rows = []
    for player, group in df.groupby("Player Name"):
        group = group.sort_values("Date", kind="stable")
        latest = group.iloc[-1]
        history = group["Average"].to_numpy(dtype=float)
        history = history[~np.isnan(history)]
        threshold = np.quantile(history, 1 - percentile) if len(history) else np.nan
        baseline = float(np.median(history)) if len(history) else np.nan
        drop = baseline - float(latest["Average"]) if len(history) else np.nan
        z = latest.get("Z-Score", np.nan)
        rows.append({
            "Player Name": player,
            "CMJ Latest": float(latest["Average"]),
            "CMJ Date": latest["Date"],
            "CMJ Threshold": threshold,
            "CMJ Baseline": baseline,
            "CMJ Drop": drop,
            "CMJ Detectable": detectable,
            "CMJ Z-Score": float(z) if pd.notna(z) else np.nan,
            "CMJ Tests": int(len(history)),
            "CMJ Fatigued": bool(
                len(history) >= MIN_HISTORY
                and latest["Average"] <= threshold
                and pd.notna(drop)
                and drop > detectable
            ),
        })
    return pd.DataFrame(rows)


def gps_state(daily, window_days=WINDOW_DAYS, percentile=PERCENTILE):
    """Per player: is her current rolling load above her own historical 75th
    percentile on either metric? The percentile is taken over that player's
    PRIOR windows only, so today's spike cannot inflate the bar it is being
    measured against.

    A current window holding a partial capture still flags when it is over the
    bar -- the recorded load alone clears it -- and reports "GPS Partial" so the
    board does not read the opposite outcome as clear."""
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
        # Only fully-populated, fully-recorded windows form the reference
        # distribution.
        complete = group["Window Complete"].to_numpy(dtype=bool)
        partial = group["Window Partial"].to_numpy(dtype=bool)
        reference = complete & ~partial
        record["GPS Partial"] = bool(partial[-1])
        triggers, prior_counts = [], []
        for metric in LOAD_METRICS:
            series = group[f"{metric} ({window_days}d)"].to_numpy(dtype=float)
            current = series[-1]
            prior = series[:-1][reference[:-1]]
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
        board["Note"] = board["Player Name"].map(lambda p: active_note(p) or "")
        board["Photo"] = board["Player Name"].map(roster.image_path)
        return board.reset_index(drop=True)

    board = cmj.merge(gps, on="Player Name", how="outer")
    # eq(True) rather than fillna(False).astype(bool): the outer merge leaves an
    # object column, and fillna's silent downcast of one is deprecated. eq gives
    # a real bool column in one step, with a missing flag reading as not-flagged.
    board["CMJ Fatigued"] = board["CMJ Fatigued"].eq(True)
    board["GPS Fatigued"] = board["GPS Fatigued"].eq(True)
    board["GPS Partial"] = board["GPS Partial"].eq(True)
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
        # Under her bar on a window that is missing part of a session is not
        # evidence she is under it (see PARTIAL_CAPTURES).
        if row["GPS Partial"]:
            return "Load understated"
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
            elif pd.api.types.is_bool_dtype(board[column]):
                # A bool column cannot hold NaN, and "GPS Fatigued" is swept up
                # by the prefix above. Pandas used to widen it to object with a
                # FutureWarning, which Streamlit swallows, so this passed locally
                # and raised TypeError on the newer pandas in the deployment.
                # Blank means "not flagged" for a boolean, so say so explicitly.
                board.loc[excluded, column] = False
            else:
                board.loc[excluded, column] = np.nan
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
                    "GPS Partial": False,
                    "On Watchlist": False,
                }])],
                ignore_index=True,
            )

    # Context for reading a row, not an input to it (see PLAYER_NOTES). A
    # partial window rides along too, so a flag or a "CMJ only" built on one
    # says so on the row -- a CMJ-only player there may be a watchlist player
    # whose load went unrecorded.
    def note(row):
        notes = [active_note(row["Player Name"], as_of)]
        if row.get("GPS Partial", False):
            notes.append("Load window missing part of a session")
        return " \u00b7 ".join(n for n in notes if n)

    board["Note"] = board.apply(note, axis=1)
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

    Partial windows are left out, and so is any date where they are most of
    the squad: the partial Sep 10 capture alone would otherwise have read as load
    falling from 38% to 52% off peak, when all that fell was the recording.
    The trend then ends at the last date it can speak for.
    """
    if daily.empty:
        return pd.Series(dtype=float)
    rolled = rolling_load(daily, window_days)
    rolled = rolled[rolled["Window Complete"]]
    if rolled.empty:
        return pd.Series(dtype=float)
    recorded_share = (~rolled["Window Partial"]).groupby(rolled["Date"]).mean()
    usable_dates = recorded_share.index[recorded_share >= 0.5]
    rolled = rolled[~rolled["Window Partial"] & rolled["Date"].isin(usable_dates)]
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

"""Reader for Catapult CTR exports, which come in two shapes.

The exports through Sep 2 2026 were a bare CSV: header row first, decimal
minutes in `Duration`, one row per player. From Sep 3 2026 the same report
arrives with

  - a nine-line metadata preamble (Date, Start Time, Num Players, Logo Path,
    ...) followed by a blank line, ahead of the real header,
  - a UTF-8 BOM,
  - `Duration` as HH:MM:SS instead of decimal minutes,
  - and every player duplicated across a "Session" row and an identical
    "Auto Created Period" row.

That last one is the dangerous difference. The two rows carry byte-identical
metrics, so anything that sums periods per player -- which is the right thing
to do for a genuine multi-period export -- silently doubles a player's load
for the day. In a fatigue tool that inflates the rolling window and flags
athletes who are not actually loaded, so the duplicates are collapsed here,
once, rather than in each caller.

`read_csv` normalises both shapes to the older layout, so callers do not have
to care which export produced a file.
"""

import io
import re

import numpy as np
import pandas as pd

# Columns that identify *which* slice of a session a row describes rather than
# what was measured during it. Two rows that agree on everything except these
# are the same measurement listed twice.
_SLICE_COLUMNS = ["Period Name", "Period Number"]

_HHMMSS_RE = re.compile(r"^\s*\d{1,3}:[0-5]\d:[0-5]\d(\.\d+)?\s*$")


def _find_header_offset(text, max_scan=40):
    """Number of leading lines before the real header. The preamble is a
    two-column key/value block, so the header is the first line that actually
    names the player column."""
    for index, line in enumerate(text.splitlines()[:max_scan]):
        stripped = line.lstrip("﻿").lstrip('"').strip()
        if stripped.lower().startswith("player name"):
            return index
    return 0


def duration_to_minutes(value):
    """HH:MM:SS -> decimal minutes. Values that are already numeric pass
    through, so a mixed season of both export shapes stays comparable."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if _HHMMSS_RE.match(text):
        hours, minutes, seconds = text.split(":")
        return int(hours) * 60 + int(minutes) + float(seconds) / 60.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_csv(source):
    """Read a Catapult CTR export of either shape into a single layout.

    `source` may be a path or a file-like object (a Streamlit upload).
    """
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
    else:
        with open(source, "r", encoding="utf-8-sig", errors="replace") as handle:
            raw = handle.read()
    raw = raw.lstrip("﻿")

    df = pd.read_csv(io.StringIO(raw), skiprows=_find_header_offset(raw))
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]

    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].map(duration_to_minutes)

    df = collapse_duplicate_periods(df)
    return df.reset_index(drop=True)


def collapse_duplicate_periods(df):
    """Drop rows that repeat a measurement under a second period label.

    Deliberately compares the measurement columns rather than trusting the
    period name: a real multi-period export has genuinely different numbers
    per period and must survive intact so callers can still sum it. Only rows
    that agree on every measured value are treated as the same row twice.

    Where a duplicate pair exists the "Session" row is the one kept, since it
    is the export's own name for the whole capture.
    """
    if "Player Name" not in df.columns or df.empty:
        return df

    measured = [c for c in df.columns if c not in _SLICE_COLUMNS]
    if len(measured) < 2:
        return df

    # Prefer the "Session" row when collapsing a duplicate pair.
    if "Period Name" in df.columns:
        priority = (df["Period Name"].astype(str).str.strip().str.lower() != "session")
        df = df.assign(_priority=priority.astype(int)).sort_values(
            "_priority", kind="stable"
        )
    else:
        df = df.assign(_priority=0)

    deduped = df.drop_duplicates(subset=measured, keep="first")
    return deduped.drop(columns="_priority").sort_index()

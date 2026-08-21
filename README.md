# Women's Soccer Sport Science Dashboard

A Streamlit dashboard with two independent tabs:

1. **CMJ Readiness** — daily pre-practice countermovement jump testing
2. **GPS / Catapult** — session load & movement data

Each tab has its own **library** of sample datasets (dropdown) and its own
**file uploader**. They are fully independent — picking a library file or
uploading in one tab never affects the other.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app2.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Folder structure

```
soccer_dashboard/
├── app2.py                    # main dashboard app
├── requirements.txt
└── sample_data/
    ├── CMJ_Preseason_Week1-2.xlsx
    ├── CMJ_InSeason_Week1-2.xlsx
    ├── CMJ_ThisWeek.xlsx
    ├── GPS_Training_Aug18.xlsx
    ├── GPS_Match_vs_Rivals.xlsx
    └── GPS_Training_Recovery.xlsx
```

## Uploading your own data

### CMJ Readiness tab expects these columns:
`Date, Match, Player Name, CMJ 1, CMJ 2, Average, Rolling Baseline,
Readiness Score, Consecutive Days, Difference, Z-Score, % Change, Rolling SD`

Only `Date, Match, Player Name, CMJ 1, CMJ 2` are actually required — if the
rest aren't in the file, the app derives them automatically
(`derive_cmj_metrics()` in `app2.py`), replicating the club's original
tracking-sheet formulas: an expanding per-player baseline/SD until a player
has 30 tests, then a trailing 30-day window; Z-Score, % Change, and a
Readiness Score/Consecutive-Days streak built from that baseline.

### GPS / Catapult tab expects these columns:
`Player Name, Period Name, Period Number, Max Acceleration, Max Deceleration,
Acceleration Efforts, Deceleration Efforts, Accel + Decel Efforts, Accel +
Decel Efforts Per Minute, Duration, Distance, Player Load, Max Velocity, Max
Vel (% Max), Meterage Per Minute, Player Load Per Minute, Work/Rest Ratio,
Max Heart Rate, Avg Heart Rate, Max HR (% Max), Avg HR (% Max), HR Exertion,
Red Zone, Heart Rate Band 1-6 Duration, Energy, High Metabolic Load Distance,
Standing/Walking/Jogging/Running/HI/Sprint/High Speed Distance, Sprint
Efforts, Sprint Dist Per Min, High Speed Efforts, High Speed Distance Per
Minute, Impacts, Athlete Tags, Activity Tags, Game Tags, Athlete
Participation Tags, Period Tags`

If your file is missing expected columns, the dashboard will still load it
and warn you which charts may not render.

## Adding more sample library entries

Drop additional matching `.xlsx` files into `sample_data/` and add an entry
to the `CMJ_LIBRARY` / `GPS_LIBRARY` dictionaries at the top of `app2.py`.

## Customizing thresholds

The CMJ readiness flag thresholds (Low / Below Average / Normal / High) are
set in the `flag_readiness()` function in `app2.py` — currently based on
Z-Score cutoffs of -1.5 / -0.5 / +0.5. Adjust to match your club's protocol.

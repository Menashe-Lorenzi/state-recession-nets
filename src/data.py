"""Data loading for the state co-movement project.

Responsibilities:
- load_states(): read Phil Fed State Coincident Indexes xls -> monthly DataFrame (50 states + DC)
- compute_returns(): pct-change monthly returns per state column
- load_nber(): monthly USREC series from the group project's daily USRECD file
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

STATES_XLS = RAW / "coincident-revised.xls"
# Monthly NBER recession indicator pulled directly from FRED (public CSV, no API key).
# The group project's USRECD only covers 2021+, so we can't use it for this project.
USREC_CSV = RAW / "USREC.csv"


def load_states() -> pd.DataFrame:
    """Load the Phil Fed State Coincident Indexes sheet 'Indexes'.

    Returns a monthly DataFrame indexed by Date, columns are state two-letter codes.
    """
    df = pd.read_excel(STATES_XLS, sheet_name="Indexes")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def compute_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Monthly simple returns: (x_t - x_{t-1}) / x_{t-1}. Drops the first NaN row.

    Index is snapped to month-end so it aligns with the FRED USREC (also snapped).
    """
    rets = levels.pct_change().dropna(how="all")
    rets.index = rets.index + pd.offsets.MonthEnd(0)
    return rets


def load_nber_monthly() -> pd.Series:
    """Load FRED's monthly NBER recession indicator USREC.

    FRED publishes USREC dated on the first of each month. We align to month-end
    to match the returns index (pandas default for resampling returns).
    """
    df = pd.read_csv(USREC_CSV, parse_dates=["observation_date"])
    df = df.set_index("observation_date").sort_index()
    s = df["USREC"].astype(int)
    # Shift from month-start to month-end timestamp so USREC aligns with returns.
    s.index = s.index + pd.offsets.MonthEnd(0)
    s.name = "USREC"
    return s

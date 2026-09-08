"""Shared helpers for the indicator library."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("open", "high", "low", "close")


_CANONICAL = ("open", "high", "low", "close", "volume")


def _is_canonical(df: pd.DataFrame) -> bool:
    """True when the frame is already normalised, so no copy is needed.

    This matters: a single signal evaluation calls ensure_ohlcv ~75 times, and
    an unconditional DataFrame copy on each made it the largest single cost in
    the backtester.
    """
    columns = df.columns
    if not all(c in columns for c in _CANONICAL):
        return False
    try:
        return all(pd.api.types.is_numeric_dtype(df[c]) for c in _CANONICAL)
    except (KeyError, TypeError):
        return False


def ensure_ohlcv(df: pd.DataFrame, need_volume: bool = False) -> pd.DataFrame:
    """Normalise column names and validate that the frame is usable.

    Accepts the common capitalisations produced by yfinance / NSE bhavcopy /
    broker APIs and returns a lower-cased frame.  Callers treat the result as
    read-only -- when the input is already canonical it is returned as-is
    rather than copied.
    """
    if _is_canonical(df):
        return df

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    alias = {
        "adj_close": "adj_close", "last": "close", "ltp": "close",
        "vol": "volume", "qty": "volume", "traded_volume": "volume",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    }
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing} (have {list(out.columns)})")
    if need_volume and "volume" not in out.columns:
        raise ValueError("this indicator needs a 'volume' column")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    for c in (*REQUIRED, "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range."""
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def wilder_smooth(s: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (RMA) -- an EMA with alpha = 1/length.

    Using this rather than a plain EMA matters: RSI/ADX/ATR values differ
    materially between the two and every published Indian-market level (e.g.
    "RSI 60 on the daily") assumes Wilder.
    """
    return s.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where ``a`` crosses above ``b``."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def slope(s: pd.Series, length: int = 5) -> pd.Series:
    """Least-squares slope over a rolling window, in units per bar."""
    x = np.arange(length, dtype=float)
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()

    def _fit(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        return float((x_centered * (window - window.mean())).sum() / denom)

    return s.rolling(length).apply(_fit, raw=True)


def percent_rank(s: pd.Series, length: int) -> pd.Series:
    """Rolling percentile rank of the latest value, in [0, 100]."""
    return s.rolling(length).apply(
        lambda w: float((w[:-1] < w[-1]).mean() * 100.0) if not np.isnan(w).any() else np.nan,
        raw=True,
    )


def zscore(s: pd.Series, length: int) -> pd.Series:
    mean = s.rolling(length).mean()
    std = s.rolling(length).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)

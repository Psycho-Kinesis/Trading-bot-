"""Volatility, range and band indicators.

Volatility state is what decides whether you should be *buying* options
(cheap IV, expansion expected) or *selling* them (rich IV, contraction
expected).  These feed the options layer as much as the direction layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import ensure_ohlcv, percent_rank, series, true_range, wilder_smooth
from .trend import ema, sma

__all__ = [
    "atr",
    "atr_percentile",
    "bollinger",
    "chandelier_exit",
    "choppiness",
    "donchian",
    "gap_stats",
    "garman_klass_volatility",
    "historical_volatility",
    "keltner",
    "natr",
    "parkinson_volatility",
    "range_expansion",
    "squeeze",
    "ulcer_index",
    "yang_zhang_volatility",
]


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return wilder_smooth(true_range(ensure_ohlcv(df)), length)


def natr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR as a percentage of price -- comparable across indices and eras."""
    d = ensure_ohlcv(df)
    return 100 * atr(d, length) / d["close"]


def bollinger(s: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(s, length)
    sd = series(s).rolling(length).std(ddof=0)
    upper, lower = mid + std * sd, mid - std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series(s) - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper, "bb_mid": mid, "bb_lower": lower,
        "bb_width": width, "bb_pct_b": pct_b,
        "bb_width_pctile": percent_rank(width, min(len(width), 252)),
    })


def keltner(df: pd.DataFrame, length: int = 20, mult: float = 2.0, atr_length: int = 10) -> pd.DataFrame:
    d = ensure_ohlcv(df)
    mid = ema(d["close"], length)
    band = mult * atr(d, atr_length)
    return pd.DataFrame({"kc_upper": mid + band, "kc_mid": mid, "kc_lower": mid - band})


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    d = ensure_ohlcv(df)
    upper = d["high"].rolling(length).max()
    lower = d["low"].rolling(length).min()
    return pd.DataFrame({"dc_upper": upper, "dc_mid": (upper + lower) / 2, "dc_lower": lower})


def squeeze(df: pd.DataFrame, bb_length: int = 20, bb_std: float = 2.0,
            kc_length: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """TTM-style squeeze: Bollinger Bands inside Keltner Channels.

    A squeeze that has persisted for several bars is the single best precursor
    to a range expansion, which is when long straddles/strangles pay.
    """
    d = ensure_ohlcv(df)
    bb = bollinger(d["close"], bb_length, bb_std)
    kc = keltner(d, kc_length, kc_mult)
    on = (bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])

    # Momentum inside the squeeze tells you which way it is likely to fire.
    mid = (d["high"].rolling(kc_length).max() + d["low"].rolling(kc_length).min()) / 2
    base = (mid + sma(d["close"], kc_length)) / 2
    detrended = d["close"] - base
    x = np.arange(kc_length, dtype=float)
    xc = x - x.mean()
    denom = (xc**2).sum()
    mom = detrended.rolling(kc_length).apply(
        lambda w: float((xc * (w - w.mean())).sum() / denom * (kc_length - 1) + w.mean())
        if not np.isnan(w).any() else np.nan,
        raw=True,
    )

    # Count how many consecutive bars the squeeze has been on.
    grp = (~on).cumsum()
    bars_on = on.groupby(grp).cumsum().where(on, 0)
    return pd.DataFrame({
        "squeeze_on": on,
        "squeeze_fired": (~on) & on.shift(1).fillna(False),
        "squeeze_bars": bars_on,
        # On the bar a squeeze fires, squeeze_bars has already reset to 0. The
        # length of the compression that just ended is what actually matters,
        # so carry the prior count forward.
        "squeeze_bars_prior": bars_on.shift(1).fillna(0),
        "squeeze_momentum": mom,
    })


def historical_volatility(s: pd.Series, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised close-to-close realised volatility, in percent.

    Compare against India VIX: HV well below VIX means options are expensive
    relative to what the index is actually delivering (favour selling).
    """
    log_ret = np.log(series(s) / series(s).shift(1))
    return log_ret.rolling(length).std(ddof=1) * np.sqrt(periods_per_year) * 100


def parkinson_volatility(df: pd.DataFrame, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    """High-low estimator -- ~5x more efficient than close-to-close."""
    d = ensure_ohlcv(df)
    hl = np.log(d["high"] / d["low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * hl.rolling(length).mean() * periods_per_year) * 100


def garman_klass_volatility(df: pd.DataFrame, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    d = ensure_ohlcv(df)
    hl = 0.5 * np.log(d["high"] / d["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(d["close"] / d["open"]) ** 2
    return np.sqrt((hl - co).rolling(length).mean() * periods_per_year) * 100


def yang_zhang_volatility(df: pd.DataFrame, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Yang-Zhang -- handles overnight gaps, which matter a lot for NIFTY where
    a large share of the move happens between 15:30 and 09:15."""
    d = ensure_ohlcv(df)
    log_ho = np.log(d["high"] / d["open"])
    log_lo = np.log(d["low"] / d["open"])
    log_co = np.log(d["close"] / d["open"])
    log_oc = np.log(d["open"] / d["close"].shift(1))

    close_vol = log_oc.rolling(length).var(ddof=1)
    open_vol = log_co.rolling(length).var(ddof=1)
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(length).mean()
    k = 0.34 / (1.34 + (length + 1) / (length - 1))
    return np.sqrt((close_vol + k * open_vol + (1 - k) * rs) * periods_per_year) * 100


def choppiness(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Choppiness Index, 0-100.  >61.8 = ranging, <38.2 = trending.

    The cleanest single filter for "do not take trend signals right now".
    """
    d = ensure_ohlcv(df)
    tr_sum = true_range(d).rolling(length).sum()
    rng = d["high"].rolling(length).max() - d["low"].rolling(length).min()
    return 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(length)


def ulcer_index(s: pd.Series, length: int = 14) -> pd.Series:
    """Downside-only volatility -- penalises drawdown depth *and* duration."""
    s = series(s)
    roll_max = s.rolling(length).max()
    drawdown = 100 * (s - roll_max) / roll_max
    return np.sqrt((drawdown**2).rolling(length).mean())


def chandelier_exit(df: pd.DataFrame, length: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """ATR trailing stop anchored to the highest high / lowest low."""
    d = ensure_ohlcv(df)
    a = atr(d, length)
    return pd.DataFrame({
        "chandelier_long": d["high"].rolling(length).max() - mult * a,
        "chandelier_short": d["low"].rolling(length).min() + mult * a,
    })


def range_expansion(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Today's range divided by the average range.  >1.5 = expansion bar."""
    d = ensure_ohlcv(df)
    rng = d["high"] - d["low"]
    return rng / rng.rolling(length).mean().replace(0, np.nan)


def atr_percentile(df: pd.DataFrame, length: int = 14, lookback: int = 252) -> pd.Series:
    """Where current ATR sits in its own history -- the volatility regime gauge."""
    return percent_rank(natr(df, length), lookback)


def gap_stats(df: pd.DataFrame, length: int = 60) -> pd.DataFrame:
    """Overnight gap analysis.

    NIFTY gaps carry a lot of the index's total move; ``gap_atr`` (the gap in
    ATR units) is the number that decides whether a gap-fade is worth taking.
    """
    d = ensure_ohlcv(df)
    prev_close = d["close"].shift(1)
    gap = d["open"] - prev_close
    gap_pct = 100 * gap / prev_close
    a = atr(d, 14)
    filled = np.where(
        gap > 0, d["low"] <= prev_close,
        np.where(gap < 0, d["high"] >= prev_close, False),
    )
    filled_series = pd.Series(filled, index=d.index)
    return pd.DataFrame({
        "gap": gap,
        "gap_pct": gap_pct,
        "gap_atr": gap / a.replace(0, np.nan),
        "gap_filled": filled_series,
        "gap_fill_rate": filled_series.rolling(length).mean(),
    })

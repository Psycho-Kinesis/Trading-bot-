"""Indicator library.

Everything is pure numpy/pandas -- no TA-Lib or other compiled dependency, so
the package installs anywhere Python does.  Two entry points matter:

* :func:`compute_all` -- attach a wide feature frame to an OHLCV frame.
* :func:`snapshot`    -- collapse the latest bar into a flat readable dict.
"""

from __future__ import annotations

import pandas as pd

from . import momentum, pivots, trend, volatility, volume
from ._util import (
    crossover,
    crossunder,
    ensure_ohlcv,
    percent_rank,
    slope,
    true_range,
    wilder_smooth,
    zscore,
)
from .momentum import (
    awesome_oscillator,
    cci,
    cmo,
    connors_rsi,
    coppock,
    ppo,
    roc,
    rsi,
    rsi_divergence,
    stoch_rsi,
    stochastic,
    tsi,
    ultimate_oscillator,
    williams_r,
)
from .momentum import momentum as mom
from .pivots import (
    camarilla_pivots,
    classic_pivots,
    cpr,
    fibonacci_pivots,
    nearest_levels,
    woodie_pivots,
)
from .trend import (
    adx,
    alma,
    aroon,
    dema,
    dpo,
    ema,
    hma,
    ichimoku,
    kama,
    ma_ribbon_score,
    macd,
    mass_index,
    psar,
    sma,
    supertrend,
    tema,
    trix,
    vortex,
    vwma,
    wma,
)
from .volatility import (
    atr,
    atr_percentile,
    bollinger,
    chandelier_exit,
    choppiness,
    donchian,
    gap_stats,
    garman_klass_volatility,
    historical_volatility,
    keltner,
    natr,
    parkinson_volatility,
    range_expansion,
    squeeze,
    ulcer_index,
    yang_zhang_volatility,
)
from .volume import (
    adl,
    anchored_vwap,
    cmf,
    ease_of_movement,
    force_index,
    has_usable_volume,
    klinger,
    mfi,
    obv,
    pvt,
    relative_volume,
    volume_profile,
    volume_zscore,
    vwap,
    vwap_bands,
)

__all__ = [
    "adl",
    "adx",
    "alma",
    "anchored_vwap",
    "aroon",
    "atr",
    "atr_percentile",
    "awesome_oscillator",
    "bollinger",
    "camarilla_pivots",
    "cci",
    "chandelier_exit",
    "choppiness",
    "classic_pivots",
    "cmf",
    "cmo",
    "compute_all",
    "connors_rsi",
    "coppock",
    "cpr",
    "crossover",
    "crossunder",
    "dema",
    "donchian",
    "dpo",
    "ease_of_movement",
    "ema",
    "ensure_ohlcv",
    "fibonacci_pivots",
    "force_index",
    "gap_stats",
    "garman_klass_volatility",
    "has_usable_volume",
    "historical_volatility",
    "hma",
    "ichimoku",
    "kama",
    "keltner",
    "klinger",
    "ma_ribbon_score",
    "macd",
    "mass_index",
    "mfi",
    "mom",
    "momentum",
    "natr",
    "nearest_levels",
    "obv",
    "parkinson_volatility",
    "percent_rank",
    "pivots",
    "ppo",
    "psar",
    "pvt",
    "range_expansion",
    "relative_volume",
    "roc",
    "rsi",
    "rsi_divergence",
    "slope",
    "sma",
    "snapshot",
    "squeeze",
    "stoch_rsi",
    "stochastic",
    "supertrend",
    "tema",
    "trend",
    "trix",
    "true_range",
    "tsi",
    "ulcer_index",
    "ultimate_oscillator",
    "volatility",
    "volume",
    "volume_profile",
    "volume_zscore",
    "vortex",
    "vwap",
    "vwap_bands",
    "vwma",
    "wilder_smooth",
    "williams_r",
    "wma",
    "woodie_pivots",
    "yang_zhang_volatility",
    "zscore",
]


def compute_all(df: pd.DataFrame, with_volume: bool | None = None, with_pivots: bool = True) -> pd.DataFrame:
    """Attach the standard indicator suite to an OHLCV frame.

    ``with_volume=None`` auto-detects whether the feed carries usable volume
    (index spot feeds frequently do not) and silently skips the volume family
    when it does not, rather than filling the frame with garbage.
    """
    d = ensure_ohlcv(df)
    out = d.copy()
    close = d["close"]

    if with_volume is None:
        with_volume = has_usable_volume(d)

    # --- trend ---------------------------------------------------------
    for n in (5, 9, 20, 21, 50, 100, 200):
        out[f"ema_{n}"] = ema(close, n)
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(close, n)
    out["hma_21"] = hma(close, 21)
    out["kama_10"] = kama(close, 10)
    out["ribbon"] = ma_ribbon_score(close)
    out = out.join(macd(close))
    out = out.join(adx(d))
    out = out.join(supertrend(d, 10, 3.0).add_prefix("st10_"))
    out = out.join(supertrend(d, 7, 2.0).add_prefix("st7_"))
    out = out.join(psar(d))
    out = out.join(ichimoku(d))
    out = out.join(aroon(d))
    out = out.join(vortex(d))
    out = out.join(trix(close))

    # --- momentum ------------------------------------------------------
    out["rsi_14"] = rsi(close, 14)
    out["rsi_7"] = rsi(close, 7)
    out["rsi_slope"] = slope(out["rsi_14"], 5)
    out = out.join(stoch_rsi(close))
    out = out.join(stochastic(d))
    out["cci_20"] = cci(d, 20)
    out["roc_10"] = roc(close, 10)
    out["williams_r"] = williams_r(d)
    out["uo"] = ultimate_oscillator(d)
    out["ao"] = awesome_oscillator(d)
    out = out.join(tsi(close))
    out["cmo_14"] = cmo(close, 14)
    out = out.join(ppo(close))
    out = out.join(rsi_divergence(d)[["bullish_divergence", "bearish_divergence"]])

    # --- volatility ----------------------------------------------------
    out["atr_14"] = atr(d, 14)
    out["natr_14"] = natr(d, 14)
    out["atr_pctile"] = atr_percentile(d)
    out = out.join(bollinger(close))
    out = out.join(keltner(d))
    out = out.join(donchian(d, 20))
    out = out.join(squeeze(d))
    out["hv_20"] = historical_volatility(close, 20)
    out["yz_vol_20"] = yang_zhang_volatility(d, 20)
    out["parkinson_20"] = parkinson_volatility(d, 20)
    out["choppiness_14"] = choppiness(d, 14)
    out["ulcer_14"] = ulcer_index(close, 14)
    out = out.join(chandelier_exit(d))
    out["range_expansion"] = range_expansion(d)
    out = out.join(gap_stats(d))

    # --- volume --------------------------------------------------------
    if with_volume:
        out["obv"] = obv(d)
        out["cmf_20"] = cmf(d, 20)
        out["mfi_14"] = mfi(d, 14)
        out["adl"] = adl(d)
        out["vwap"] = vwap(d)
        out = out.join(vwap_bands(d)[["vwap_sd", "vwap_upper_1", "vwap_lower_1",
                                      "vwap_upper_2", "vwap_lower_2"]])
        out["rel_volume"] = relative_volume(d)
        out["volume_z"] = volume_zscore(d)
        out["force_index"] = force_index(d)
        out["obv_slope"] = slope(out["obv"], 10)

    # --- pivots (daily-bar frames only) --------------------------------
    if with_pivots:
        out = out.join(cpr(d))
        out = out.join(classic_pivots(d).add_prefix("pp_"))
        out = out.join(camarilla_pivots(d)[["h3", "l3", "h4", "l4"]].add_prefix("cam_"))

    # ``attrs`` does not survive joins, so stamp it last.
    out.attrs["volume_available"] = bool(with_volume)
    return out


def snapshot(features: pd.DataFrame, index: int = -1) -> dict:
    """Flatten one bar of a computed feature frame into a plain dict.

    NaNs become ``None`` so the result is JSON-serialisable and safe to log.
    """
    import numpy as np

    row = features.iloc[index]
    out: dict = {}
    for key, value in row.items():
        key = str(key)
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
            out[key] = None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = round(float(value), 4)
        else:
            out[key] = None if pd.isna(value) else value
    ts = features.index[index]
    out["timestamp"] = str(ts)
    return out

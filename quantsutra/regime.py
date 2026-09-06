"""Market regime classification.

The single most useful thing a trading system can do is refuse to apply a
trend strategy in a range and a mean-reversion strategy in a trend.  Every
playbook in :mod:`quantsutra.signals.playbooks` declares which regimes it is
valid in, and the engine hard-gates on this classification.

The classifier is deliberately built from *orthogonal* inputs -- trend
strength (ADX), directional alignment (MA ribbon), range-vs-trend character
(Choppiness), and volatility level (ATR percentile) -- rather than several
restatements of the same momentum number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import Regime
from .indicators._util import ensure_ohlcv
from .indicators.momentum import rsi
from .indicators.trend import adx, ema, ma_ribbon_score
from .indicators.volatility import (atr_percentile, choppiness, historical_volatility,
                                    natr, squeeze)

__all__ = ["RegimeState", "classify_regime", "volatility_state"]


@dataclass
class RegimeState:
    regime: Regime
    trend_score: float          # -1 (down) .. +1 (up)
    trend_strength: float       # 0..1, how directional the tape is
    volatility_state: str       # LOW | NORMAL | HIGH | EXTREME
    adx: float
    choppiness: float
    atr_percentile: float
    natr: float
    squeeze_bars: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in (Regime.STRONG_UPTREND, Regime.UPTREND,
                               Regime.DOWNTREND, Regime.STRONG_DOWNTREND)

    @property
    def is_bullish(self) -> bool:
        return self.regime in (Regime.STRONG_UPTREND, Regime.UPTREND)

    @property
    def is_bearish(self) -> bool:
        return self.regime in (Regime.STRONG_DOWNTREND, Regime.DOWNTREND)

    @property
    def favors_option_buying(self) -> bool:
        """Buying premium needs either a strong trend or an imminent expansion."""
        return (self.trend_strength > 0.55 and self.is_trending) or self.regime == Regime.SQUEEZE

    @property
    def favors_option_selling(self) -> bool:
        """Selling premium needs range-bound price and elevated implied vol."""
        return self.regime in (Regime.RANGE,) and self.volatility_state in ("HIGH", "EXTREME")

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "trend_score": round(self.trend_score, 3),
            "trend_strength": round(self.trend_strength, 3),
            "volatility_state": self.volatility_state,
            "adx": round(self.adx, 1),
            "choppiness": round(self.choppiness, 1),
            "atr_percentile": round(self.atr_percentile, 1),
            "natr": round(self.natr, 3),
            "squeeze_bars": self.squeeze_bars,
            "is_trending": self.is_trending,
            "favors_option_buying": self.favors_option_buying,
            "favors_option_selling": self.favors_option_selling,
            "notes": self.notes,
        }


def volatility_state(atr_pct: float, india_vix: float | None = None) -> str:
    """Bucket volatility, preferring India VIX when it is available.

    India VIX bands used here (approximate, long-run): below 12 is complacent,
    12-16 normal, 16-22 elevated, above 22 is event/panic territory.  Verify
    against the current distribution -- these bands drift across decades.
    """
    if india_vix is not None and np.isfinite(india_vix):
        if india_vix < 12:
            return "LOW"
        if india_vix < 16:
            return "NORMAL"
        if india_vix < 22:
            return "HIGH"
        return "EXTREME"
    if not np.isfinite(atr_pct):
        return "NORMAL"
    if atr_pct < 25:
        return "LOW"
    if atr_pct < 70:
        return "NORMAL"
    if atr_pct < 90:
        return "HIGH"
    return "EXTREME"


def classify_regime(df: pd.DataFrame, india_vix: float | None = None) -> RegimeState:
    """Classify the current regime from an OHLCV frame.

    Needs roughly 60 bars to be meaningful; with less it returns RANGE with a
    low trend strength rather than guessing.
    """
    d = ensure_ohlcv(df)
    notes: list[str] = []
    if len(d) < 60:
        notes.append("fewer than 60 bars: regime is unreliable, defaulting to RANGE")
        return RegimeState(Regime.RANGE, 0.0, 0.0, "NORMAL", np.nan, np.nan, np.nan,
                           np.nan, 0, notes)

    close = d["close"]
    adx_df = adx(d, 14)
    adx_val = float(adx_df["adx"].iloc[-1])
    plus_di = float(adx_df["plus_di"].iloc[-1])
    minus_di = float(adx_df["minus_di"].iloc[-1])
    chop = float(choppiness(d, 14).iloc[-1])
    atr_pct = float(atr_percentile(d).iloc[-1])
    natr_val = float(natr(d, 14).iloc[-1])
    ribbon = float(ma_ribbon_score(close).iloc[-1])
    rsi_val = float(rsi(close, 14).iloc[-1])
    sq = squeeze(d)
    squeeze_bars = int(sq["squeeze_bars"].iloc[-1]) if not pd.isna(sq["squeeze_bars"].iloc[-1]) else 0

    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])
    last = float(close.iloc[-1])

    # --- direction: combine several orthogonal votes into [-1, 1] --------
    votes = []
    if np.isfinite(ribbon):
        votes.append(ribbon)
    if np.isfinite(plus_di) and np.isfinite(minus_di) and (plus_di + minus_di) > 0:
        votes.append((plus_di - minus_di) / (plus_di + minus_di))
    if np.isfinite(rsi_val):
        votes.append(np.clip((rsi_val - 50) / 25, -1, 1))
    if np.isfinite(ema20) and np.isfinite(ema50) and ema50 > 0:
        votes.append(np.clip((ema20 - ema50) / (0.01 * ema50), -1, 1))
    if np.isfinite(last) and np.isfinite(ema50) and ema50 > 0:
        votes.append(np.clip((last - ema50) / (0.02 * ema50), -1, 1))
    trend_score = float(np.mean(votes)) if votes else 0.0

    # --- strength: how much to believe that direction --------------------
    adx_component = np.clip((adx_val - 15) / 25, 0, 1) if np.isfinite(adx_val) else 0.0
    chop_component = np.clip((61.8 - chop) / 25, 0, 1) if np.isfinite(chop) else 0.0
    align_component = abs(trend_score)
    trend_strength = float(np.clip(
        0.45 * adx_component + 0.30 * chop_component + 0.25 * align_component, 0, 1
    ))

    vol_state = volatility_state(atr_pct, india_vix)

    # --- assemble --------------------------------------------------------
    if squeeze_bars >= 4 and trend_strength < 0.55:
        regime = Regime.SQUEEZE
        notes.append(f"volatility compressed for {squeeze_bars} bars: expansion is due, "
                     "favour long premium / straddles over directional bets")
    elif np.isfinite(chop) and chop > 61.8 and vol_state in ("HIGH", "EXTREME"):
        regime = Regime.VOLATILE_CHOP
        notes.append("high volatility with no direction: the worst environment for "
                     "both trend-following and naked option selling")
    elif trend_strength >= 0.65 and trend_score > 0.25:
        regime = Regime.STRONG_UPTREND
    elif trend_strength >= 0.65 and trend_score < -0.25:
        regime = Regime.STRONG_DOWNTREND
    elif trend_strength >= 0.40 and trend_score > 0.15:
        regime = Regime.UPTREND
    elif trend_strength >= 0.40 and trend_score < -0.15:
        regime = Regime.DOWNTREND
    else:
        regime = Regime.RANGE

    if np.isfinite(adx_val) and adx_val < 20 and regime in (Regime.UPTREND, Regime.DOWNTREND):
        notes.append(f"ADX {adx_val:.0f} is below 20: the trend label is weak, size down")
    if np.isfinite(chop) and chop > 61.8 and regime != Regime.VOLATILE_CHOP:
        notes.append(f"Choppiness {chop:.0f} says ranging: breakout signals will whipsaw")
    if vol_state == "EXTREME":
        notes.append("extreme volatility: widen stops or stand aside; option premiums "
                     "are rich but tail risk is real")
    if vol_state == "LOW" and regime == Regime.RANGE:
        notes.append("low volatility range: option selling is theoretically favoured but "
                     "premiums are thin -- costs eat the edge")

    return RegimeState(
        regime=regime, trend_score=trend_score, trend_strength=trend_strength,
        volatility_state=vol_state, adx=adx_val, choppiness=chop,
        atr_percentile=atr_pct, natr=natr_val, squeeze_bars=squeeze_bars, notes=notes,
    )

"""The rule library.

Each rule reads a :class:`~quantsutra.signals.base.MarketContext` and returns a
:class:`~quantsutra.signals.base.RuleResult` scored in [-1, +1].  Rules are
grouped into families and the engine caps each family's contribution, so
adding a ninth trend rule does not make the engine nine-eighths as confident
about the trend.

A rule that cannot evaluate (missing data, not enough history) must return
``confidence=0``.  The engine drops those instead of counting them as neutral
votes, which would otherwise wash out real signals.
"""

from __future__ import annotations

import numpy as np

from .base import Category, MarketContext, RuleResult

__all__ = ["ALL_RULES", "evaluate_rules"]


def _nil(name: str, category: str, why: str) -> RuleResult:
    return RuleResult(name, category, 0.0, confidence=0.0, rationale=why)


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


# ---------------------------------------------------------------------------
# TREND
# ---------------------------------------------------------------------------

def rule_ma_stack(ctx: MarketContext) -> RuleResult:
    """Alignment of the whole moving-average ribbon."""
    ribbon = ctx.get("ribbon")
    if ribbon is None:
        return _nil("ma_stack", Category.TREND, "moving-average ribbon unavailable")
    if ribbon > 0.6:
        why = f"EMA ribbon fully stacked bullish ({ribbon:+.2f}): every average is above the next."
    elif ribbon < -0.6:
        why = f"EMA ribbon fully stacked bearish ({ribbon:+.2f})."
    else:
        why = f"EMA ribbon is tangled ({ribbon:+.2f}) -- no clean trend alignment."
    return RuleResult("ma_stack", Category.TREND, _clip(ribbon), weight=1.2,
                      rationale=why, detail={"ribbon": ribbon})


def rule_price_vs_emas(ctx: MarketContext) -> RuleResult:
    """Where price sits relative to the 20/50/200 EMAs."""
    close = ctx.get("close")
    e20, e50, e200 = ctx.get("ema_20"), ctx.get("ema_50"), ctx.get("ema_200")
    votes, parts = [], []
    for name, level in (("20EMA", e20), ("50EMA", e50), ("200EMA", e200)):
        if level is None or close is None:
            continue
        votes.append(1.0 if close > level else -1.0)
        parts.append(f"{'above' if close > level else 'below'} {name}")
    if not votes:
        return _nil("price_vs_emas", Category.TREND, "EMAs unavailable")
    score = float(np.mean(votes))
    return RuleResult("price_vs_emas", Category.TREND, score, weight=1.0,
                      rationale=f"Price is {', '.join(parts)}.")


def rule_adx(ctx: MarketContext) -> RuleResult:
    """ADX/DMI: trend strength and which side owns it."""
    adx = ctx.get("adx")
    plus, minus = ctx.get("plus_di"), ctx.get("minus_di")
    if adx is None or plus is None or minus is None:
        return _nil("adx_dmi", Category.TREND, "ADX unavailable")
    if adx < 20:
        return RuleResult("adx_dmi", Category.TREND, 0.0, weight=1.1, confidence=0.4,
                          rationale=f"ADX {adx:.0f} is below 20 -- no trend to follow, "
                                    f"directional signals will chop.")
    direction = 1.0 if plus > minus else -1.0
    strength = float(np.clip((adx - 20) / 25, 0, 1))
    score = direction * strength
    side = "+DI over -DI" if direction > 0 else "-DI over +DI"
    return RuleResult("adx_dmi", Category.TREND, score, weight=1.3,
                      rationale=f"ADX {adx:.0f} with {side}: a genuine "
                                f"{'up' if direction > 0 else 'down'}trend is in force.")


def rule_supertrend(ctx: MarketContext) -> RuleResult:
    """Supertrend direction on two settings; agreement raises confidence."""
    fast, slow = ctx.get("st7_direction"), ctx.get("st10_direction")
    if fast is None and slow is None:
        return _nil("supertrend", Category.TREND, "Supertrend unavailable")
    votes = [v for v in (fast, slow) if v is not None]
    score = float(np.mean(votes))
    agree = len(votes) == 2 and fast == slow
    flip = bool(ctx.get("st10_flip") or ctx.get("st7_flip"))
    why = ("Both Supertrends are " + ("long" if score > 0 else "short")) if agree else \
          "Supertrend settings disagree -- the trend is transitioning"
    if flip:
        why += "; a flip printed on this bar, which is the earliest entry but also the "\
               "most whipsaw-prone."
    return RuleResult("supertrend", Category.TREND, score, weight=1.1,
                      confidence=1.0 if agree else 0.6, rationale=why + ".")


def rule_macd(ctx: MarketContext) -> RuleResult:
    macd, signal, hist = ctx.get("macd"), ctx.get("signal"), ctx.get("hist")
    if macd is None or signal is None or hist is None:
        return _nil("macd", Category.TREND, "MACD unavailable")
    atr = ctx.atr if ctx.atr and np.isfinite(ctx.atr) else abs(ctx.spot) * 0.005
    normalised = _clip(hist / (atr * 0.5)) if atr > 0 else 0.0
    above = macd > signal
    zero = "above" if macd > 0 else "below"
    return RuleResult("macd", Category.TREND, normalised, weight=1.0,
                      rationale=f"MACD is {'above' if above else 'below'} its signal and "
                                f"{zero} zero (histogram {hist:+.1f}).")


def rule_ichimoku(ctx: MarketContext) -> RuleResult:
    close = ctx.get("close")
    a, b = ctx.get("senkou_a"), ctx.get("senkou_b")
    tenkan, kijun = ctx.get("tenkan"), ctx.get("kijun")
    if close is None or a is None or b is None:
        return _nil("ichimoku", Category.TREND, "Ichimoku cloud unavailable")
    top, bottom = max(a, b), min(a, b)
    if close > top:
        cloud, cloud_score = "above the cloud", 1.0
    elif close < bottom:
        cloud, cloud_score = "below the cloud", -1.0
    else:
        cloud, cloud_score = "inside the cloud (no-trade zone)", 0.0
    tk = 0.0
    if tenkan is not None and kijun is not None:
        tk = 1.0 if tenkan > kijun else -1.0
    score = 0.7 * cloud_score + 0.3 * tk
    return RuleResult("ichimoku", Category.TREND, score, weight=0.9,
                      confidence=0.6 if cloud_score == 0 else 1.0,
                      rationale=f"Price is {cloud}; Tenkan is "
                                f"{'above' if tk > 0 else 'below'} Kijun.")


# ---------------------------------------------------------------------------
# MOMENTUM
# ---------------------------------------------------------------------------

def rule_rsi(ctx: MarketContext) -> RuleResult:
    """RSI read in a regime-aware way.

    In a trend, RSI>70 is *strength*, not a sell.  In a range it is a fade.
    Reading overbought as bearish inside a strong uptrend is the classic way
    to be run over, so the sign flips on the regime.
    """
    rsi = ctx.get("rsi_14")
    if rsi is None:
        return _nil("rsi", Category.MOMENTUM, "RSI unavailable")
    trending = ctx.regime.is_trending and ctx.regime.trend_strength > 0.5

    if trending:
        score = _clip((rsi - 50) / 25)
        if rsi > 70:
            why = f"RSI {rsi:.0f} is overbought, but in a confirmed trend that is momentum, not a sell."
        elif rsi < 30:
            why = f"RSI {rsi:.0f} is oversold in a downtrend -- momentum, not a buy."
        else:
            why = f"RSI {rsi:.0f} leans {'bullish' if rsi > 50 else 'bearish'}."
        return RuleResult("rsi", Category.MOMENTUM, score, weight=1.0, rationale=why)

    if rsi > 70:
        return RuleResult("rsi", Category.MOMENTUM, -0.7, weight=1.1,
                          rationale=f"RSI {rsi:.0f} overbought in a range -- fade territory.")
    if rsi < 30:
        return RuleResult("rsi", Category.MOMENTUM, 0.7, weight=1.1,
                          rationale=f"RSI {rsi:.0f} oversold in a range -- bounce territory.")
    return RuleResult("rsi", Category.MOMENTUM, _clip((rsi - 50) / 40), weight=0.7,
                      rationale=f"RSI {rsi:.0f} is mid-range -- no momentum edge.")


def rule_rsi_divergence(ctx: MarketContext) -> RuleResult:
    bull = bool(ctx.get("bullish_divergence"))
    bear = bool(ctx.get("bearish_divergence"))
    if not (bull or bear):
        return _nil("rsi_divergence", Category.MOMENTUM, "no RSI divergence")
    if bull:
        return RuleResult("rsi_divergence", Category.MOMENTUM, 0.8, weight=1.2,
                          rationale="Bullish RSI divergence: price made a lower low while "
                                    "RSI made a higher low -- selling pressure is fading.")
    return RuleResult("rsi_divergence", Category.MOMENTUM, -0.8, weight=1.2,
                      rationale="Bearish RSI divergence: price made a higher high while "
                                "RSI made a lower high -- the rally is losing fuel.")


def rule_stochastic(ctx: MarketContext) -> RuleResult:
    k, d = ctx.get("stoch_k"), ctx.get("stoch_d")
    if k is None or d is None:
        return _nil("stochastic", Category.MOMENTUM, "Stochastic unavailable")
    cross = 1.0 if k > d else -1.0
    extreme = 0.0
    if k > 80:
        extreme = -0.5 if not ctx.regime.is_trending else 0.3
    elif k < 20:
        extreme = 0.5 if not ctx.regime.is_trending else -0.3
    score = _clip(0.6 * cross + 0.4 * extreme)
    return RuleResult("stochastic", Category.MOMENTUM, score, weight=0.7,
                      rationale=f"Stochastic %K {k:.0f} is "
                                f"{'above' if cross > 0 else 'below'} %D.")


def rule_momentum_pack(ctx: MarketContext) -> RuleResult:
    """CCI + Williams %R + ROC + TSI as one combined momentum vote."""
    votes = []
    parts = []
    cci = ctx.get("cci_20")
    if cci is not None:
        votes.append(_clip(cci / 150))
        parts.append(f"CCI {cci:+.0f}")
    wr = ctx.get("williams_r")
    if wr is not None:
        votes.append(_clip((wr + 50) / 40))
        parts.append(f"W%R {wr:.0f}")
    roc = ctx.get("roc_10")
    if roc is not None:
        votes.append(_clip(roc / 2.5))
        parts.append(f"ROC {roc:+.1f}%")
    tsi = ctx.get("tsi")
    if tsi is not None:
        votes.append(_clip(tsi / 30))
        parts.append(f"TSI {tsi:+.0f}")
    if not votes:
        return _nil("momentum_pack", Category.MOMENTUM, "momentum oscillators unavailable")
    score = float(np.mean(votes))
    return RuleResult("momentum_pack", Category.MOMENTUM, score, weight=0.9,
                      rationale=f"Momentum oscillators ({', '.join(parts)}) net "
                                f"{'bullish' if score > 0 else 'bearish' if score < 0 else 'flat'}.")


# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------

def rule_market_structure(ctx: MarketContext) -> RuleResult:
    st = ctx.structure
    if st is None:
        return _nil("market_structure", Category.STRUCTURE, "structure unavailable")
    base = {"UPTREND": 1.0, "DOWNTREND": -1.0}.get(st.trend, 0.0)
    score = base * st.strength
    why = f"Structure is {st.trend} ({st.label})"
    if st.last_event:
        why += f"; last event {st.last_event}"
        if st.bars_since_event is not None:
            why += f" {st.bars_since_event} bars ago"
    return RuleResult("market_structure", Category.STRUCTURE, score, weight=1.5,
                      rationale=why + ".", detail=st.to_dict())


def rule_break_of_structure(ctx: MarketContext) -> RuleResult:
    """A fresh BOS/CHoCH is the single highest-value structure event."""
    st = ctx.structure
    if st is None or not st.last_event:
        return _nil("break_of_structure", Category.STRUCTURE, "no recent structure break")
    bars = st.bars_since_event if st.bars_since_event is not None else 99
    if bars > 5:
        return _nil("break_of_structure", Category.STRUCTURE,
                    f"last structure break was {bars} bars ago -- stale")
    freshness = 1.0 - bars / 6
    up = st.last_event.endswith("_UP")
    is_choch = st.last_event.startswith("CHOCH")
    score = (1.0 if up else -1.0) * freshness * (0.9 if is_choch else 1.0)
    if is_choch:
        why = (f"Change of character {bars} bars ago: the first break against the prior "
               f"trend, which is the earliest warning that it has ended.")
    else:
        why = (f"Break of structure {bars} bars ago in the direction of the trend -- "
               f"continuation confirmed.")
    return RuleResult("break_of_structure", Category.STRUCTURE, score, weight=1.4,
                      rationale=why)


def rule_liquidity_sweep(ctx: MarketContext) -> RuleResult:
    sweeps = [s for s in (ctx.snapshot.get("_sweeps") or []) if s.get("bars_ago", 99) <= 4]
    if not sweeps:
        return _nil("liquidity_sweep", Category.STRUCTURE, "no recent liquidity sweep")
    latest = sweeps[0]
    freshness = 1.0 - latest["bars_ago"] / 5
    magnitude = min(1.0, latest.get("wick_atr", 0.5) / 1.5)
    score = latest["bias"] * freshness * (0.5 + 0.5 * magnitude)
    return RuleResult("liquidity_sweep", Category.STRUCTURE, score, weight=1.3,
                      rationale=f"{latest['kind'].replace('_', ' ').title()} at "
                                f"{latest['level']:.0f} {latest['bars_ago']} bars ago: "
                                f"{latest['note']}.")


def rule_fair_value_gap(ctx: MarketContext) -> RuleResult:
    gaps = [g for g in (ctx.snapshot.get("_fvgs") or []) if g.get("bars_ago", 99) <= 12]
    if not gaps:
        return _nil("fair_value_gap", Category.STRUCTURE, "no unfilled fair value gaps nearby")
    latest = gaps[0]
    direction = 1.0 if latest["kind"] == "BULLISH" else -1.0
    size = min(1.0, latest.get("size_atr", 0.5) / 1.5)
    unfilled = 1.0 - latest.get("filled_pct", 0.0)
    score = direction * size * unfilled * 0.8
    return RuleResult("fair_value_gap", Category.STRUCTURE, score, weight=0.9,
                      rationale=f"Unfilled {latest['kind'].lower()} fair value gap "
                                f"{latest['bottom']:.0f}-{latest['top']:.0f} "
                                f"({latest['bars_ago']} bars ago) -- price tends to revisit "
                                f"these, and they act as support/resistance on the way.")


# ---------------------------------------------------------------------------
# PATTERN
# ---------------------------------------------------------------------------

def rule_candlestick(ctx: MarketContext) -> RuleResult:
    if not ctx.candles:
        return _nil("candlestick", Category.PATTERN, "no candlestick pattern on the last bar")
    scored = []
    names = []
    for sig in ctx.candles:
        if sig.bias == 0:
            continue
        weight = sig.strength * (1.0 if sig.context_ok else 0.35)
        scored.append(sig.bias * weight)
        names.append(f"{sig.name}{'' if sig.context_ok else ' (weak context)'}")
    if not scored:
        return _nil("candlestick", Category.PATTERN, "only indecision candles present")
    score = _clip(float(np.mean(scored)))
    return RuleResult("candlestick", Category.PATTERN, score, weight=1.0,
                      rationale=f"Candlestick: {', '.join(names[:3])}.")


def rule_chart_pattern(ctx: MarketContext) -> RuleResult:
    if not ctx.chart_patterns:
        return _nil("chart_pattern", Category.PATTERN, "no classical chart pattern")
    actionable = [p for p in ctx.chart_patterns if p.bias != 0]
    if not actionable:
        return _nil("chart_pattern", Category.PATTERN,
                    "only neutral formations (symmetrical triangle / range)")
    best = max(actionable, key=lambda p: p.confidence * (1.3 if p.status == "CONFIRMED" else 1.0))
    score = best.bias * best.confidence * (1.0 if best.status == "CONFIRMED" else 0.5)
    tgt = f", measured target {best.target:.0f}" if best.target else ""
    return RuleResult("chart_pattern", Category.PATTERN, _clip(score), weight=1.2,
                      rationale=f"{best.name.replace('_', ' ').title()} "
                                f"({best.status.lower()}), trigger {best.trigger:.0f}{tgt}.",
                      detail=best.to_dict())


# ---------------------------------------------------------------------------
# LEVELS
# ---------------------------------------------------------------------------

def rule_level_proximity(ctx: MarketContext) -> RuleResult:
    """Room to the next level in each direction.

    Being 0.3 ATR below heavy resistance is a bad place to buy no matter how
    bullish everything else looks -- this rule is what stops the engine from
    recommending entries into a wall.
    """
    info = ctx.snapshot.get("_levels_info")
    if not info:
        return _nil("level_proximity", Category.LEVELS, "no level map available")
    up = info.get("room_to_resistance_atr")
    down = info.get("room_to_support_atr")
    if up is None and down is None:
        return _nil("level_proximity", Category.LEVELS, "no levels above or below")
    up = up if up is not None else 5.0
    down = down if down is not None else 5.0
    # Positive when there is more room above than below.
    score = _clip((up - down) / 4)
    parts = []
    if info.get("nearest_resistance"):
        parts.append(f"resistance {info['nearest_resistance']:.0f} ({up:.1f} ATR up)")
    if info.get("nearest_support"):
        parts.append(f"support {info['nearest_support']:.0f} ({down:.1f} ATR down)")
    why = "Room: " + "; ".join(parts) + "."
    if info.get("at_level"):
        why += " Price is sitting ON a level -- wait for acceptance or rejection."
    return RuleResult("level_proximity", Category.LEVELS, score, weight=1.2, rationale=why)


def rule_cpr(ctx: MarketContext) -> RuleResult:
    """Central Pivot Range position and width."""
    close = ctx.get("close")
    top, bottom = ctx.get("cpr_top"), ctx.get("cpr_bottom")
    cpr_type = ctx.snapshot.get("cpr_type")
    if close is None or top is None or bottom is None:
        return _nil("cpr", Category.LEVELS, "CPR unavailable (needs daily bars)")
    if close > top:
        pos, score = "above the CPR", 0.7
    elif close < bottom:
        pos, score = "below the CPR", -0.7
    else:
        pos, score = "inside the CPR", 0.0
    why = f"Price is {pos}"
    conf = 1.0
    if cpr_type == "NARROW":
        why += "; the CPR is narrow, which historically favours a trending day"
        score *= 1.2
    elif cpr_type == "WIDE":
        why += "; the CPR is wide, which favours a range-bound day"
        score *= 0.6
        conf = 0.7
    if ctx.get("cpr_higher_value"):
        why += "; today's CPR sits entirely above yesterday's (bullish value migration)"
        score = max(score, 0.4)
    elif ctx.get("cpr_lower_value"):
        why += "; today's CPR sits entirely below yesterday's (bearish value migration)"
        score = min(score, -0.4)
    return RuleResult("cpr", Category.LEVELS, _clip(score), weight=1.0, confidence=conf,
                      rationale=why + ".")


def rule_vwap(ctx: MarketContext) -> RuleResult:
    close, vwap = ctx.get("close"), ctx.get("vwap")
    if close is None or vwap is None:
        return _nil("vwap", Category.LEVELS, "VWAP unavailable (needs volume)")
    sd = ctx.get("vwap_sd")
    if sd and sd > 0:
        z = (close - vwap) / sd
        if abs(z) > 2 and not ctx.regime.is_trending:
            return RuleResult("vwap", Category.LEVELS, _clip(-z / 3), weight=1.1,
                              rationale=f"Price is {z:+.1f} SD from VWAP in a range -- "
                                        f"stretched, mean reversion is the higher-probability "
                                        f"trade.")
        return RuleResult("vwap", Category.LEVELS, _clip(z / 2), weight=1.0,
                          rationale=f"Price is {z:+.1f} SD {'above' if z > 0 else 'below'} "
                                    f"VWAP -- {'buyers' if z > 0 else 'sellers'} in control "
                                    f"of the session.")
    score = 0.6 if close > vwap else -0.6
    return RuleResult("vwap", Category.LEVELS, score, weight=1.0,
                      rationale=f"Price is {'above' if score > 0 else 'below'} VWAP.")


# ---------------------------------------------------------------------------
# VOLUME
# ---------------------------------------------------------------------------

def rule_volume_confirmation(ctx: MarketContext) -> RuleResult:
    rel = ctx.get("rel_volume")
    close, open_ = ctx.get("close"), ctx.get("open")
    if rel is None or close is None or open_ is None:
        return _nil("volume_confirmation", Category.VOLUME, "volume unavailable")
    bar_dir = 1.0 if close > open_ else (-1.0 if close < open_ else 0.0)
    if rel < 0.8:
        return RuleResult("volume_confirmation", Category.VOLUME, bar_dir * 0.1,
                          weight=0.8, confidence=0.5,
                          rationale=f"Volume is {rel:.1f}x average -- thin participation, "
                                    f"the move lacks conviction.")
    strength = min(1.0, (rel - 0.8) / 1.7)
    return RuleResult("volume_confirmation", Category.VOLUME, bar_dir * strength, weight=1.0,
                      rationale=f"Volume {rel:.1f}x average confirms the "
                                f"{'up' if bar_dir > 0 else 'down'} bar.")


def rule_money_flow(ctx: MarketContext) -> RuleResult:
    cmf, mfi, obv_slope = ctx.get("cmf_20"), ctx.get("mfi_14"), ctx.get("obv_slope")
    votes, parts = [], []
    if cmf is not None:
        votes.append(_clip(cmf / 0.2))
        parts.append(f"CMF {cmf:+.2f}")
    if mfi is not None:
        votes.append(_clip((mfi - 50) / 25))
        parts.append(f"MFI {mfi:.0f}")
    if obv_slope is not None and ctx.atr and ctx.atr > 0:
        votes.append(_clip(float(np.sign(obv_slope))) * 0.5)
        parts.append(f"OBV {'rising' if obv_slope > 0 else 'falling'}")
    if not votes:
        return _nil("money_flow", Category.VOLUME, "money-flow indicators unavailable")
    score = float(np.mean(votes))
    return RuleResult("money_flow", Category.VOLUME, score, weight=1.0,
                      rationale=f"Money flow ({', '.join(parts)}) is "
                                f"{'accumulating' if score > 0.15 else 'distributing' if score < -0.15 else 'neutral'}.")


# ---------------------------------------------------------------------------
# VOLATILITY
# ---------------------------------------------------------------------------

def rule_squeeze(ctx: MarketContext) -> RuleResult:
    """A squeeze is directionless by itself -- it says a move is coming."""
    on = bool(ctx.get("squeeze_on"))
    bars = ctx.get("squeeze_bars") or 0
    fired = bool(ctx.get("squeeze_fired"))
    mom = ctx.get("squeeze_momentum")
    if fired and mom is not None:
        score = _clip(np.sign(mom) * 0.6)
        return RuleResult("squeeze", Category.VOLATILITY, score, weight=1.3,
                          rationale=f"Squeeze just fired {'upward' if mom > 0 else 'downward'} "
                                    f"-- volatility is expanding out of compression, which is "
                                    f"the highest-quality breakout context.")
    if on and bars >= 4:
        return RuleResult("squeeze", Category.VOLATILITY, 0.0, weight=1.0, confidence=0.5,
                          rationale=f"Volatility has been compressed for {int(bars)} bars. "
                                    f"A range expansion is due but the direction is not yet "
                                    f"decided -- favour a straddle over a directional bet.")
    return _nil("squeeze", Category.VOLATILITY, "no volatility squeeze")


def rule_volatility_regime(ctx: MarketContext) -> RuleResult:
    """Volatility level is not directional, but it gates position size."""
    state = ctx.regime.volatility_state
    atr_pct = ctx.regime.atr_percentile
    if state == "EXTREME":
        return RuleResult("volatility_regime", Category.VOLATILITY, 0.0, weight=1.0,
                          confidence=0.6,
                          rationale=f"Volatility is extreme (ATR percentile {atr_pct:.0f}). "
                                    f"Stops must be far wider, so the same rupee risk buys a "
                                    f"much smaller position.")
    if state == "LOW":
        return RuleResult("volatility_regime", Category.VOLATILITY, 0.0, weight=0.8,
                          confidence=0.4,
                          rationale=f"Volatility is low (ATR percentile {atr_pct:.0f}) -- "
                                    f"option premiums are cheap but moves may be small.")
    return _nil("volatility_regime", Category.VOLATILITY, "volatility is unremarkable")


def rule_gap(ctx: MarketContext) -> RuleResult:
    gap_atr = ctx.get("gap_atr")
    fill_rate = ctx.get("gap_fill_rate")
    if gap_atr is None or abs(gap_atr) < 0.4:
        return _nil("gap", Category.VOLATILITY, "no significant opening gap")
    direction = np.sign(gap_atr)
    if fill_rate is not None and fill_rate > 0.6 and abs(gap_atr) < 1.5:
        return RuleResult("gap", Category.VOLATILITY, _clip(-direction * 0.4), weight=0.9,
                          rationale=f"Gapped {'up' if direction > 0 else 'down'} "
                                    f"{abs(gap_atr):.1f} ATR; {fill_rate*100:.0f}% of recent "
                                    f"gaps this size filled, so a fade has the edge.")
    return RuleResult("gap", Category.VOLATILITY, _clip(direction * 0.5), weight=0.9,
                      rationale=f"Large {'up' if direction > 0 else 'down'} gap "
                                f"({abs(gap_atr):.1f} ATR) -- these tend to run rather than "
                                f"fill; treat it as a trend day until proven otherwise.")


# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------

def rule_pcr(ctx: MarketContext) -> RuleResult:
    cs = ctx.chain_summary
    if not cs or cs.get("pcr_oi") is None:
        return _nil("pcr", Category.OPTIONS, "option chain unavailable")
    pcr = cs["pcr_oi"]
    # Mildly contrarian at the extremes, mildly confirmatory in between.
    if pcr > 1.6:
        score, why = -0.3, (f"PCR {pcr:.2f} is extreme: put writing is crowded, which "
                            f"usually precedes a squeeze *against* the crowd.")
    elif pcr > 1.1:
        score, why = 0.5, f"PCR {pcr:.2f}: put writers are confident, supportive of upside."
    elif pcr > 0.8:
        score, why = 0.0, f"PCR {pcr:.2f} is balanced -- no positioning edge."
    elif pcr > 0.55:
        score, why = -0.5, f"PCR {pcr:.2f}: call writing dominates, capping rallies."
    else:
        score, why = 0.3, (f"PCR {pcr:.2f} is extremely low: call writing is crowded and "
                           f"vulnerable to a short-covering squeeze.")
    return RuleResult("pcr", Category.OPTIONS, score, weight=1.0, rationale=why)


def rule_oi_flow(ctx: MarketContext) -> RuleResult:
    cs = ctx.chain_summary
    if not cs:
        return _nil("oi_flow", Category.OPTIONS, "option chain unavailable")
    bias = cs.get("oi_flow_bias")
    if not bias or bias == "BALANCED":
        return _nil("oi_flow", Category.OPTIONS, "today's OI additions are balanced")
    if bias == "BULLISH_PUT_WRITING":
        return RuleResult("oi_flow", Category.OPTIONS, 0.6, weight=1.1,
                          rationale="Today's OI is being added mostly on the put side: "
                                    "writers are selling downside, which is bullish "
                                    "positioning and creates support at those strikes.")
    return RuleResult("oi_flow", Category.OPTIONS, -0.6, weight=1.1,
                      rationale="Today's OI is being added mostly on the call side: "
                                "writers are capping the upside, which is bearish positioning.")


def rule_oi_walls(ctx: MarketContext) -> RuleResult:
    """Distance to the heaviest call/put OI strikes."""
    cs = ctx.chain_summary
    if not cs or cs.get("max_call_oi_strike") is None or cs.get("max_put_oi_strike") is None:
        return _nil("oi_walls", Category.OPTIONS, "OI walls unavailable")
    spot = ctx.spot
    call_wall = cs["max_call_oi_strike"]
    put_wall = cs["max_put_oi_strike"]
    up_room = (call_wall - spot) / spot * 100
    down_room = (spot - put_wall) / spot * 100
    if up_room <= 0 or down_room <= 0:
        return _nil("oi_walls", Category.OPTIONS,
                    "price has traded through an OI wall -- the level is no longer meaningful")
    score = _clip((up_room - down_room) / 1.5)
    return RuleResult("oi_walls", Category.OPTIONS, score, weight=0.9,
                      rationale=f"Heaviest call OI at {call_wall:.0f} ({up_room:.2f}% above) "
                                f"and put OI at {put_wall:.0f} ({down_room:.2f}% below) -- "
                                f"the expected range for this expiry.")


def rule_iv_context(ctx: MarketContext) -> RuleResult:
    """IV level does not give direction, but it decides *how* to express one."""
    ivp = ctx.iv_percentile
    vix = ctx.india_vix
    if ivp is None and vix is None:
        return _nil("iv_context", Category.OPTIONS, "no IV history available")
    parts = []
    if vix is not None:
        parts.append(f"India VIX {vix:.1f}")
    if ivp is not None:
        parts.append(f"IV percentile {ivp:.0f}")
    if ivp is not None and ivp > 75:
        why = (f"{', '.join(parts)}: options are expensive. Buying premium here needs the "
               f"move to be fast; a spread or a credit structure is usually better.")
    elif ivp is not None and ivp < 25:
        why = (f"{', '.join(parts)}: options are cheap. Long premium is favoured and "
               f"the IV downside is limited.")
    else:
        why = f"{', '.join(parts)}: implied volatility is around its normal level."
    return RuleResult("iv_context", Category.OPTIONS, 0.0, weight=1.0, confidence=0.5,
                      rationale=why)


# ---------------------------------------------------------------------------
# CONTEXT
# ---------------------------------------------------------------------------

def rule_higher_timeframe(ctx: MarketContext) -> RuleResult:
    """Trading against the higher timeframe is the most expensive habit there is."""
    htf = ctx.higher_tf
    if not htf:
        return _nil("higher_timeframe", Category.CONTEXT, "no higher-timeframe context")
    ribbon = htf.get("ribbon")
    close, ema50 = htf.get("close"), htf.get("ema_50")
    votes = []
    if ribbon is not None and np.isfinite(ribbon):
        votes.append(float(ribbon))
    if close is not None and ema50 is not None and np.isfinite(close) and np.isfinite(ema50):
        votes.append(1.0 if close > ema50 else -1.0)
    if not votes:
        return _nil("higher_timeframe", Category.CONTEXT, "higher-timeframe data incomplete")
    score = float(np.mean(votes))
    tf = htf.get("_timeframe", "higher timeframe")
    return RuleResult("higher_timeframe", Category.CONTEXT, score, weight=1.4,
                      rationale=f"The {tf} trend is "
                                f"{'up' if score > 0.2 else 'down' if score < -0.2 else 'flat'} "
                                f"-- taking trades against it cuts the win rate materially.")


def rule_session_phase(ctx: MarketContext) -> RuleResult:
    """Some phases of the Indian session are structurally hostile."""
    phase = ctx.session_phase
    if phase == "OPENING_AUCTION_DRIFT":
        return RuleResult("session_phase", Category.CONTEXT, 0.0, weight=1.0, confidence=0.5,
                          rationale="First 15 minutes: spreads are widest, direction is "
                                    "unreliable and stop hunts are common. Most edges do not "
                                    "exist here.")
    if phase == "MIDDAY":
        return RuleResult("session_phase", Category.CONTEXT, 0.0, weight=0.8, confidence=0.4,
                          rationale="Midday lull (10:15-12:30): the lowest-volume window of "
                                    "the Indian session, where breakouts most often fail.")
    if phase == "CLOSING_HOUR":
        return RuleResult("session_phase", Category.CONTEXT, 0.0, weight=0.8, confidence=0.3,
                          rationale="Final hour: directional moves are real but option "
                                    "premium decays fastest here, especially near expiry.")
    return _nil("session_phase", Category.CONTEXT, f"session phase {phase} is unremarkable")


def rule_expiry_proximity(ctx: MarketContext) -> RuleResult:
    exp = ctx.expiry
    if not exp:
        return _nil("expiry_proximity", Category.CONTEXT, "no expiry context")
    dte = exp.get("dte_trading")
    if dte is None:
        return _nil("expiry_proximity", Category.CONTEXT, "days-to-expiry unknown")
    if exp.get("is_expiry_day"):
        return RuleResult("expiry_proximity", Category.CONTEXT, 0.0, weight=1.2, confidence=0.6,
                          rationale="EXPIRY DAY: premium collapses toward intrinsic through "
                                    "the session and gamma is extreme. Directional buying "
                                    "needs a fast move; naked selling is dangerous.")
    if dte <= 1:
        return RuleResult("expiry_proximity", Category.CONTEXT, 0.0, weight=1.0, confidence=0.5,
                          rationale=f"{dte} trading day to expiry: theta is at its steepest. "
                                    f"An option bought here loses value fast if the move "
                                    f"does not come immediately.")
    return _nil("expiry_proximity", Category.CONTEXT,
                f"{dte} trading days to expiry -- adequate time value")


def rule_max_pain(ctx: MarketContext) -> RuleResult:
    """Weak evidence, and only near expiry -- see chain.max_pain."""
    cs = ctx.chain_summary
    exp = ctx.expiry or {}
    if not cs or cs.get("max_pain") is None:
        return _nil("max_pain", Category.OPTIONS, "max pain unavailable")
    dte = exp.get("dte_trading", 5)
    if dte is None or dte > 1:
        return _nil("max_pain", Category.OPTIONS,
                    "max pain only carries (weak) information in the last day or two")
    distance = cs.get("max_pain_distance_pct") or 0.0
    score = _clip(distance / 1.0) * 0.4
    return RuleResult("max_pain", Category.OPTIONS, score, weight=0.5, confidence=0.4,
                      rationale=f"Max pain sits at {cs['max_pain']:.0f} ({distance:+.2f}% from "
                                f"spot). This is weak evidence -- it matters only as a mild "
                                f"pull on a quiet expiry day.")


ALL_RULES = [
    # trend
    rule_ma_stack, rule_price_vs_emas, rule_adx, rule_supertrend, rule_macd, rule_ichimoku,
    # momentum
    rule_rsi, rule_rsi_divergence, rule_stochastic, rule_momentum_pack,
    # structure
    rule_market_structure, rule_break_of_structure, rule_liquidity_sweep, rule_fair_value_gap,
    # pattern
    rule_candlestick, rule_chart_pattern,
    # levels
    rule_level_proximity, rule_cpr, rule_vwap,
    # volume
    rule_volume_confirmation, rule_money_flow,
    # volatility
    rule_squeeze, rule_volatility_regime, rule_gap,
    # options
    rule_pcr, rule_oi_flow, rule_oi_walls, rule_iv_context, rule_max_pain,
    # context
    rule_higher_timeframe, rule_session_phase, rule_expiry_proximity,
]


def evaluate_rules(ctx: MarketContext, rules=None) -> list[RuleResult]:
    """Run every rule, isolating failures so one bad rule cannot kill a scan."""
    results = []
    for rule in (rules or ALL_RULES):
        try:
            results.append(rule(ctx))
        except Exception as exc:  # a broken rule must not take down the engine
            results.append(RuleResult(getattr(rule, "__name__", "unknown"), Category.CONTEXT,
                                      0.0, confidence=0.0,
                                      rationale=f"rule raised {type(exc).__name__}: {exc}"))
    return results

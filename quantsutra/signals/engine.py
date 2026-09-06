"""The signal engine: context assembly, confluence scoring, and veto logic.

Scoring works in three stages:

1. **Per-category aggregation.**  Rules inside a family are averaged (weighted),
   so eight trend rules produce one trend opinion.
2. **Capped combination.**  Each family contributes at most its cap from
   :data:`~quantsutra.signals.base.CATEGORY_CAPS`, and the total is normalised
   by the caps that actually had data.  This is what stops correlated
   indicators from manufacturing false confidence.
3. **Vetoes.**  Hard blocks that no amount of confluence can override -- a
   trade against a strong higher timeframe, entry into a wall, a chain that is
   too illiquid to exit, a regime the setup does not work in.

The output confidence is a *relative* measure of how much the evidence agrees,
not a probability of profit.  It has not been calibrated against realised
outcomes; do that yourself with the backtester before sizing on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..calendar_in import expiry_context, session_phase
from ..constants import IST, Action, Direction, Regime
from ..indicators import compute_all, snapshot
from ..patterns.candles import detect_candles
from ..patterns.classical import detect_chart_patterns
from ..patterns.levels import (confluence_zones, find_levels, level_interaction,
                               reinforce_with_round_numbers, round_number_levels)
from ..patterns.structure import (fair_value_gaps, liquidity_sweeps, market_structure,
                                  order_blocks)
from ..regime import classify_regime
from .base import CATEGORY_CAPS, Category, MarketContext, RuleResult, Signal
from .rules import ALL_RULES, evaluate_rules

__all__ = ["EngineConfig", "SignalEngine", "build_context"]


@dataclass
class EngineConfig:
    """Thresholds and gates.  Every number here is a choice, not a law."""

    # Minimum |score| before the engine will call a direction at all.
    min_abs_score: float = 0.22
    # Minimum confidence (0-100) before a signal is marked actionable.
    min_confidence: float = 55.0
    # Confidence above which the engine considers a setup high-conviction.
    high_conviction: float = 72.0

    # Risk geometry.
    stop_atr_mult: float = 1.5
    max_stop_atr_mult: float = 3.0        # a structural stop wider than this is not tradable
    # Targets are multiples of R (the actual risk to the stop), not of ATR.
    # Fixed ATR targets against a structural stop produce inconsistent
    # reward/risk: widen the stop and every trade silently becomes unviable.
    target_r_multiples: tuple[float, ...] = (1.5, 2.5, 4.0)
    min_reward_risk: float = 1.2

    # Vetoes.
    veto_against_higher_tf: bool = True
    veto_htf_threshold: float = 0.55      # HTF conviction needed to block a counter-trade
    veto_wall_atr: float = 0.4            # min ATR of room to the next level
    veto_in_volatile_chop: bool = True
    veto_illiquid_chain: bool = True

    # Options.
    target_delta: float = 0.42
    intended_hold_days: int = 1
    min_oi: int = 25_000

    # Data hygiene.
    min_bars: int = 120


def build_context(
    symbol: str, ohlcv: pd.DataFrame, timeframe: str = "1d",
    higher_tf_ohlcv: pd.DataFrame | None = None, higher_tf_name: str = "1wk",
    chain_summary: dict | None = None, india_vix: float | None = None,
    iv_percentile: float | None = None, now: dt.datetime | None = None,
    with_volume: bool | None = None,
) -> MarketContext:
    """Assemble everything the rules read, exactly once."""
    now = now or dt.datetime.now(IST)
    features = compute_all(ohlcv, with_volume=with_volume)
    snap = snapshot(features)

    spot = float(features["close"].iloc[-1])
    atr = float(features["atr_14"].iloc[-1]) if "atr_14" in features else float("nan")
    if not np.isfinite(atr) or atr <= 0:
        atr = spot * 0.005

    regime = classify_regime(features, india_vix=india_vix)
    structure = market_structure(features)
    candles = detect_candles(features, lookback=1)
    patterns = detect_chart_patterns(features)

    levels = find_levels(features)
    levels_all = reinforce_with_round_numbers(levels + round_number_levels(spot), atr)
    snap["_levels_info"] = level_interaction(spot, levels_all, atr)
    # Only well-established levels may veto a trade; a round number 0.3 ATR away
    # is information, not an obstacle.
    snap["_veto_levels_info"] = level_interaction(spot, levels_all, atr, min_strength=0.5)
    snap["_confluence"] = confluence_zones(levels_all, atr)
    snap["_sweeps"] = liquidity_sweeps(features)
    snap["_fvgs"] = fair_value_gaps(features)
    snap["_order_blocks"] = order_blocks(features)

    htf = None
    if higher_tf_ohlcv is not None and len(higher_tf_ohlcv) >= 60:
        htf_features = compute_all(higher_tf_ohlcv, with_volume=with_volume)
        htf = snapshot(htf_features)
        htf["_timeframe"] = higher_tf_name
        htf["_regime"] = classify_regime(htf_features).to_dict()

    try:
        exp = expiry_context(symbol, now.date())
        expiry_info = {
            "expiry": exp.expiry, "dte_trading": exp.dte_trading,
            "dte_calendar": exp.dte_calendar, "is_expiry_day": exp.is_expiry_day,
            "is_monthly": exp.is_monthly, "lot_size": exp.lot_size,
            "strike_step": exp.strike_step, "gamma_risk": exp.gamma_risk,
        }
    except (KeyError, ValueError):
        expiry_info = None

    return MarketContext(
        symbol=symbol, timeframe=timeframe, now=now, ohlcv=ohlcv, features=features,
        snapshot=snap, regime=regime, spot=spot, atr=atr, higher_tf=htf,
        chain_summary=chain_summary, india_vix=india_vix, iv_percentile=iv_percentile,
        expiry=expiry_info, levels=levels_all, candles=candles, chart_patterns=patterns,
        structure=structure, session_phase=session_phase(now),
    )


class SignalEngine:
    """Turns a :class:`MarketContext` into a :class:`Signal`."""

    def __init__(self, config: EngineConfig | None = None, rules=None):
        self.config = config or EngineConfig()
        self.rules = rules or ALL_RULES

    # -- scoring ----------------------------------------------------------
    def score(self, results: list[RuleResult]) -> tuple[float, dict]:
        """Aggregate rule results into a score in [-1, 1] plus a per-family view."""
        by_cat: dict[str, list[RuleResult]] = {}
        for r in results:
            if r.confidence <= 0:
                continue
            by_cat.setdefault(r.category, []).append(r)

        cat_scores: dict[str, float] = {}
        for cat, rules in by_cat.items():
            total_w = sum(r.weight * r.confidence for r in rules)
            if total_w <= 0:
                continue
            cat_scores[cat] = float(np.clip(
                sum(r.effective for r in rules) / total_w, -1, 1
            ))

        # Normalise by the caps that actually contributed, so a missing options
        # chain does not silently deflate every score.
        active_caps = sum(CATEGORY_CAPS.get(c, 0.05) for c in cat_scores)
        if active_caps <= 0:
            return 0.0, cat_scores
        raw = sum(cat_scores[c] * CATEGORY_CAPS.get(c, 0.05) for c in cat_scores) / active_caps
        return float(np.clip(raw, -1, 1)), cat_scores

    def confidence(self, raw: float, cat_scores: dict, results: list[RuleResult]) -> float:
        """Confidence blends magnitude, breadth and agreement.

        A score of 0.4 driven by every family agreeing is worth much more than
        the same 0.4 driven by one family screaming while the rest disagree.
        """
        if not cat_scores:
            return 0.0
        magnitude = min(1.0, abs(raw) / 0.6)

        direction = np.sign(raw)
        agreeing = sum(1 for v in cat_scores.values() if np.sign(v) == direction and abs(v) > 0.1)
        disagreeing = sum(1 for v in cat_scores.values() if np.sign(v) == -direction and abs(v) > 0.1)
        total = agreeing + disagreeing
        agreement = (agreeing / total) if total else 0.5

        breadth = min(1.0, len(cat_scores) / 6)
        active = [r for r in results if r.confidence > 0]
        data_quality = min(1.0, len(active) / 18)

        blended = 0.40 * magnitude + 0.30 * agreement + 0.15 * breadth + 0.15 * data_quality
        return float(np.clip(blended * 100, 0, 100))

    # -- vetoes -----------------------------------------------------------
    def vetoes(self, ctx: MarketContext, direction: Direction) -> list[str]:
        cfg = self.config
        out: list[str] = []
        if direction == Direction.NEUTRAL:
            return out

        bullish = direction == Direction.UP

        if len(ctx.ohlcv) < cfg.min_bars:
            out.append(f"Only {len(ctx.ohlcv)} bars of history (need {cfg.min_bars}); "
                       f"indicators are not yet reliable.")

        if cfg.veto_against_higher_tf and ctx.higher_tf:
            htf_ribbon = ctx.higher_tf.get("ribbon")
            if htf_ribbon is not None and np.isfinite(htf_ribbon):
                if bullish and htf_ribbon < -cfg.veto_htf_threshold:
                    out.append(f"Higher timeframe ({ctx.higher_tf.get('_timeframe')}) is "
                               f"firmly bearish (ribbon {htf_ribbon:+.2f}); a long here is "
                               f"counter-trend.")
                elif not bullish and htf_ribbon > cfg.veto_htf_threshold:
                    out.append(f"Higher timeframe ({ctx.higher_tf.get('_timeframe')}) is "
                               f"firmly bullish (ribbon {htf_ribbon:+.2f}); a short here is "
                               f"counter-trend.")

        info = ctx.snapshot.get("_veto_levels_info") or {}
        room_key = "room_to_resistance_atr" if bullish else "room_to_support_atr"
        room = info.get(room_key)
        if room is not None and room < cfg.veto_wall_atr:
            level = info.get("nearest_resistance" if bullish else "nearest_support")
            out.append(f"Only {room:.2f} ATR of room to the next "
                       f"{'resistance' if bullish else 'support'} at {level:.0f} -- entering "
                       f"here means buying into a wall.")

        if cfg.veto_in_volatile_chop and ctx.regime.regime == Regime.VOLATILE_CHOP:
            out.append("Regime is volatile chop: high range with no direction is where "
                       "directional systems bleed fastest.")

        if ctx.regime.regime == Regime.SQUEEZE and ctx.regime.trend_strength < 0.4:
            out.append("Volatility is compressed with no established direction; the break "
                       "could go either way. A straddle expresses this better than a "
                       "directional trade.")

        return out

    # -- risk geometry ----------------------------------------------------
    def levels_for(self, ctx: MarketContext, direction: Direction) -> dict:
        """Entry, stop and targets in index points.

        The stop comes first and everything else is derived from it.  It is the
        wider of an ATR stop and the nearest structural invalidation level,
        because a stop inside the noise band gets taken out even when the idea
        is right -- then capped, because a stop too far away is not tradable.

        Targets are R-multiples of that risk, so reward/risk is consistent by
        construction rather than an accident of how wide the stop happened to be.
        """
        cfg = self.config
        spot, atr = ctx.spot, ctx.atr
        bullish = direction == Direction.UP
        atr_stop = spot - cfg.stop_atr_mult * atr if bullish else spot + cfg.stop_atr_mult * atr

        struct_stop = None
        st = ctx.structure
        if st is not None:
            struct_stop = st.swing_low if bullish else st.swing_high

        if struct_stop is not None and np.isfinite(struct_stop):
            buffer = atr * 0.25
            candidate = struct_stop - buffer if bullish else struct_stop + buffer
            stop = min(atr_stop, candidate) if bullish else max(atr_stop, candidate)
            source = "structure" if stop == candidate else "atr"
        else:
            stop, source = atr_stop, "atr"

        max_risk = cfg.max_stop_atr_mult * atr
        capped = False
        if abs(spot - stop) > max_risk:
            stop = spot - max_risk if bullish else spot + max_risk
            capped, source = True, "capped"

        risk = abs(spot - stop)
        if risk <= 0:
            risk = atr * cfg.stop_atr_mult
            stop = spot - risk if bullish else spot + risk

        sign = 1.0 if bullish else -1.0
        targets = [spot + sign * m * risk for m in cfg.target_r_multiples]

        # If a well-established level sits between the entry and the first
        # target, that level -- not the arithmetic R-multiple -- is the realistic
        # destination.  Snapping to it is honest: it lowers the reported
        # reward/risk, and if that drops below the minimum the trade is vetoed,
        # which is the correct outcome for an entry with a wall in front of it.
        info = ctx.snapshot.get("_veto_levels_info") or {}
        wall = info.get("nearest_resistance" if bullish else "nearest_support")
        snapped_to_level = False
        if wall is not None and np.isfinite(wall):
            inside = (bullish and spot < wall < targets[0]) or (not bullish and targets[0] < wall < spot)
            if inside:
                targets[0] = float(wall)
                snapped_to_level = True

        rr = abs(targets[0] - spot) / risk if risk > 0 else None
        return {
            "entry": spot, "stop_loss": float(stop),
            "targets": [float(t) for t in targets],
            "risk_points": float(risk),
            "risk_reward": round(float(rr), 2) if rr else None,
            "stop_capped": capped, "stop_source": source,
            "target_at_level": snapped_to_level,
        }

    # -- main entry point --------------------------------------------------
    def generate(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        results = evaluate_rules(ctx, self.rules)
        raw, cat_scores = self.score(results)
        conf = self.confidence(raw, cat_scores, results)

        if raw >= cfg.min_abs_score:
            direction = Direction.UP
        elif raw <= -cfg.min_abs_score:
            direction = Direction.DOWN
        else:
            direction = Direction.NEUTRAL

        vetoes = self.vetoes(ctx, direction)
        warnings: list[str] = list(ctx.regime.notes)

        signal = Signal(
            symbol=ctx.symbol, timestamp=ctx.now, timeframe=ctx.timeframe,
            direction=direction, action=Action.NO_TRADE, confidence=conf, raw_score=raw,
            spot=ctx.spot, rules=results, category_scores=cat_scores,
            regime=ctx.regime.to_dict(), warnings=warnings, vetoes=vetoes,
        )

        if direction == Direction.NEUTRAL:
            signal.reasons = [
                f"Net score {raw:+.2f} is inside the +/-{cfg.min_abs_score} neutral band -- "
                f"the evidence does not lean far enough either way to justify a directional "
                f"trade.",
            ] + signal.top_reasons(4)
            return signal

        if conf < cfg.min_confidence:
            signal.vetoes.append(
                f"Confidence {conf:.0f} is below the {cfg.min_confidence:.0f} threshold: the "
                f"rules that fired do not agree strongly enough."
            )

        geometry = self.levels_for(ctx, direction)
        signal.entry = geometry["entry"]
        signal.stop_loss = geometry["stop_loss"]
        signal.targets = geometry["targets"]
        signal.risk_reward = geometry["risk_reward"]
        if geometry.get("stop_capped"):
            warnings.append(
                f"The structural stop sat further than {cfg.max_stop_atr_mult} ATR away, so it "
                f"was capped at {geometry['risk_points']:.0f} points. The technical invalidation "
                f"level is beyond this stop, so expect to be stopped out on noise more often."
            )

        if geometry["risk_reward"] is not None and geometry["risk_reward"] < cfg.min_reward_risk:
            because = ("a well-established level sits in the way"
                       if geometry.get("target_at_level")
                       else "the stop is too wide for the move on offer")
            signal.vetoes.append(
                f"Reward/risk to the first target is {geometry['risk_reward']:.2f}, below the "
                f"{cfg.min_reward_risk} minimum -- {because}."
            )

        if not signal.vetoes:
            signal.action = (Action.BUY_CALL if direction == Direction.UP else Action.BUY_PUT)

        signal.reasons = signal.top_reasons(7)

        if conf >= cfg.high_conviction and not signal.vetoes:
            signal.reasons.insert(0, f"High-conviction setup: {len(cat_scores)} independent "
                                     f"families agree, net score {raw:+.2f}.")
        return signal

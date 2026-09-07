"""Top-level orchestration: data in, complete trade recommendation out.

This is the function the CLI and any application should call.  It stitches
together the whole pipeline -- data validation, indicators, patterns, regime,
options chain, signal engine, structure selection, sizing and event risk --
and returns a single dictionary describing what to do and why.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import pandas as pd

from .calendar_in import expiry_context, session_phase
from .constants import IST, Action, Direction
from .data.base import validate_frame
from .knowledge.events import event_risk, lessons_for_regime
from .options.chain import OptionChain, iv_rank
from .options.selection import recommend_contract, select_expiry
from .options.strategies import build_strategy, choose_structure
from .risk.costs import CostModel
from .risk.sizing import size_option_position
from .signals.engine import EngineConfig, SignalEngine, build_context
from .signals.playbooks import match_playbooks

__all__ = ["AnalysisConfig", "analyze"]


@dataclass
class AnalysisConfig:
    capital: float = 500_000.0
    risk_pct: float = 0.01
    target_delta: float = 0.42
    intended_hold_days: int = 2
    max_premium_pct: float = 0.25
    exchange: str = "NSE"
    engine: EngineConfig = None

    def __post_init__(self):
        if self.engine is None:
            self.engine = EngineConfig()


def analyze(
    symbol: str,
    ohlcv: pd.DataFrame,
    timeframe: str = "1d",
    higher_tf: pd.DataFrame | None = None,
    higher_tf_name: str = "1wk",
    chain: OptionChain | None = None,
    india_vix: float | None = None,
    vix_history: pd.Series | None = None,
    now: dt.datetime | None = None,
    config: AnalysisConfig | None = None,
) -> dict:
    """Run the full pipeline and return a complete recommendation.

    Everything optional degrades gracefully: without an option chain you still
    get a directional view and index-level levels, just no strike selection.
    """
    cfg = config or AnalysisConfig()
    now = now or dt.datetime.now(IST)

    result: dict = {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "generated_at": now.isoformat(), "session_phase": session_phase(now),
        "data_quality": [], "disclaimer": DISCLAIMER,
    }

    # --- data hygiene first -------------------------------------------
    issues = validate_frame(ohlcv, symbol)
    result["data_quality"] = issues
    if len(ohlcv) < 60:
        result["error"] = (f"Only {len(ohlcv)} bars supplied. At least 60 are needed for "
                           f"the indicators to mean anything, and 250 for the regime "
                           f"classifier to be reliable.")
        return result

    # --- IV context ----------------------------------------------------
    iv_percentile = None
    if vix_history is not None and india_vix is not None:
        iv_percentile = iv_rank(india_vix, vix_history).get("iv_percentile")
        result["iv_context"] = iv_rank(india_vix, vix_history)
    elif chain is not None and vix_history is not None:
        atm_iv = chain.summary().get("atm_iv")
        if atm_iv:
            result["iv_context"] = iv_rank(atm_iv, vix_history)
            iv_percentile = result["iv_context"].get("iv_percentile")

    chain_summary = None
    if chain is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chain.compute_ivs()
            chain_summary = chain.summary()
        result["chain"] = chain_summary

    # --- signal --------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = build_context(symbol, ohlcv, timeframe, higher_tf_ohlcv=higher_tf,
                            higher_tf_name=higher_tf_name, chain_summary=chain_summary,
                            india_vix=india_vix, iv_percentile=iv_percentile, now=now)
        signal = SignalEngine(cfg.engine).generate(ctx)
        playbooks = match_playbooks(ctx)

    result["signal"] = signal.to_dict()
    result["playbooks"] = [p.to_dict() for p in playbooks]
    result["levels"] = _level_summary(ctx)
    result["structure"] = ctx.structure.to_dict() if ctx.structure else None
    result["candles"] = [
        {"name": c.name, "bias": c.bias, "strength": round(c.strength, 2),
         "context_ok": c.context_ok, "notes": c.notes}
        for c in ctx.candles
    ]
    result["chart_patterns"] = [p.to_dict() for p in ctx.chart_patterns]
    result["event_risk"] = event_risk(now.date(), symbol)
    result["historical_lessons"] = lessons_for_regime(
        ctx.regime.regime.value, ctx.regime.volatility_state
    )

    # --- expiry, structure choice, strike, sizing ----------------------
    try:
        exp = expiry_context(symbol, now.date())
        result["expiry"] = select_expiry(symbol, now.date(), cfg.intended_hold_days)
    except (KeyError, ValueError):
        exp = None

    if signal.direction == Direction.NEUTRAL or not signal.is_actionable:
        result["recommendation"] = _no_trade_recommendation(signal, playbooks)
        return result

    direction = 1 if signal.direction == Direction.UP else -1
    dte = (result.get("expiry") or {}).get("dte_trading", exp.dte_trading if exp else 3)
    action, structure_notes = choose_structure(
        direction, iv_percentile, dte, ctx.regime.trend_strength
    )
    signal.action = action
    result["signal"]["action"] = action.value
    result["structure_reasoning"] = structure_notes

    if chain is not None and action != Action.NO_TRADE:
        contract = recommend_contract(action, chain, cfg.target_delta)
        result["contract"] = contract
        if contract["complete"] and contract["strikes"]:
            strategy = build_strategy(action, chain.spot, contract["strikes"],
                                      contract["premiums"], chain.lot_size,
                                      contract["ivs"], chain.expiry)
            result["strategy"] = strategy.to_dict()
            result["strategy"]["greeks"] = strategy.net_greeks(chain.t)
            result["position"] = _size_from_chain(
                cfg, chain, contract, signal, action, ctx
            )
            result["costs"] = _cost_estimate(cfg, strategy, chain)
    elif chain is None:
        result["contract_note"] = (
            "No option chain supplied, so no strike could be selected. The direction, "
            "levels and risk geometry above are still valid; fetch a chain "
            "(`quantsutra chain`) to get a specific contract."
        )

    result["recommendation"] = _build_recommendation(signal, result, playbooks)
    return result


def _level_summary(ctx) -> dict:
    info = ctx.snapshot.get("_levels_info") or {}
    return {
        "nearest_support": info.get("nearest_support"),
        "nearest_resistance": info.get("nearest_resistance"),
        "room_to_support_atr": info.get("room_to_support_atr"),
        "room_to_resistance_atr": info.get("room_to_resistance_atr"),
        "at_level": info.get("at_level"),
        "confluence_zones": ctx.snapshot.get("_confluence") or [],
        "fair_value_gaps": (ctx.snapshot.get("_fvgs") or [])[:3],
        "order_blocks": (ctx.snapshot.get("_order_blocks") or [])[:3],
        "liquidity_sweeps": (ctx.snapshot.get("_sweeps") or [])[:3],
        "atr": round(ctx.atr, 2),
        "cpr": {
            "top": ctx.get("cpr_top"), "bottom": ctx.get("cpr_bottom"),
            "type": ctx.snapshot.get("cpr_type"),
        },
    }


def _size_from_chain(cfg, chain, contract, signal, action, ctx) -> dict:
    """Size the position from the resolved contract."""
    role = next((r for r in ("long_ce", "long_pe", "short_ce", "short_pe")
                 if r in contract["premiums"]), None)
    if role is None:
        return {"lots": 0, "reasons": ["No priced leg to size against."]}

    premium = contract["premiums"][role]
    strike = contract["strikes"][role]
    option_type = "CE" if role.endswith("ce") else "PE"
    from .options.pricing import greeks

    iv = (contract["ivs"].get(role) or 15.0) / 100
    g = greeks(chain.spot, strike, chain.t, iv, option_type)
    stop_distance = abs(signal.spot - signal.stop_loss) if signal.stop_loss else ctx.atr * 1.5

    sizing = size_option_position(
        capital=cfg.capital, risk_pct=cfg.risk_pct, premium=premium,
        lot_size=chain.lot_size, delta=g.delta, index_stop_distance=stop_distance,
        max_premium_pct=cfg.max_premium_pct,
        direction="SHORT" if role.startswith("short") else "LONG",
    )
    out = sizing.to_dict()
    out["premium_per_unit"] = round(premium, 2)
    out["total_premium"] = round(premium * sizing.units, 2)
    if sizing.lots < 1:
        # Report what it would actually take, rather than only that it is zero.
        risk_per_lot = max(premium * 0.40, abs(g.delta) * stop_distance)
        risk_per_lot = min(risk_per_lot, premium) * chain.lot_size
        out["capital_needed_for_one_lot"] = round(risk_per_lot / max(cfg.risk_pct, 1e-9), 2)
    return out


def _cost_estimate(cfg, strategy, chain) -> dict:
    model = CostModel(exchange=cfg.exchange)
    total = None
    for leg in strategy.legs:
        breakdown = model.option_leg(
            leg.premium, chain.lot_size * abs(leg.quantity),
            "BUY" if leg.is_long else "SELL",
            is_expiry_day=chain.dte_trading <= 0,
        )
        total = breakdown if total is None else total + breakdown
    if total is None:
        return {}
    # A round trip is entry plus exit, so double the one-way estimate.
    return {
        "one_way": total.to_dict(),
        "estimated_round_trip": round(total.total * 2, 2),
        "note": ("Round trip estimate for one lot of each leg, including an assumed "
                 "bid-ask spread. Your actual spread is the number that matters -- "
                 "check it before entering."),
    }


def _no_trade_recommendation(signal, playbooks) -> dict:
    lines = []
    if signal.vetoes:
        lines.append("Blocked by: " + "; ".join(signal.vetoes))
    else:
        lines.append(f"No directional edge: net score {signal.raw_score:+.2f} sits inside "
                     f"the neutral band.")
    watching = [p.to_dict() for p in playbooks if p.direction != Direction.NEUTRAL]
    if watching:
        lines.append(f"{len(watching)} playbook(s) are close to triggering -- see the "
                     f"playbooks section for their trigger levels.")
    return {
        "action": "NO_TRADE",
        "summary": "Stand aside.",
        "details": lines,
        "watching": watching,
    }


def _build_recommendation(signal, result, playbooks) -> dict:
    position = result.get("position") or {}
    strategy = result.get("strategy") or {}
    lines: list[str] = []

    action = result["signal"]["action"]
    if not position:
        # No chain means no priced contract, so nothing was sized. Saying "zero
        # at your risk limit" here would blame the wrong thing entirely.
        summary = (f"{action} setup identified. No option chain was supplied, so no "
                   f"strike was priced and no size was computed.")
        lines.append("The direction, levels and risk geometry are usable as they stand. "
                     "Supply a chain to get a specific contract and a lot count.")
    elif position.get("lots", 0) < 1:
        needed = position.get("capital_needed_for_one_lot")
        summary = f"{action} setup identified, but position size is ZERO at your risk limit."
        detail = ("The signal is valid; the trade is not takeable at this capital and "
                  "risk setting. Sizing up to take it anyway is how accounts break.")
        if needed:
            detail += f" One lot would need about Rs.{needed:,.0f} of capital."
        lines.append(detail)
    else:
        summary = (f"{action}: {position.get('lots')} lot(s), "
                   f"risking about Rs.{position.get('risk_amount', 0):,.0f} "
                   f"({position.get('risk_pct_of_capital', 0) * 100:.2f}% of capital).")

    if signal.entry and signal.stop_loss:
        lines.append(f"Index levels: entry {signal.entry:,.0f}, stop {signal.stop_loss:,.0f}, "
                     f"first target {signal.targets[0]:,.0f} "
                     f"(reward/risk {signal.risk_reward}).")
    if strategy.get("breakevens"):
        lines.append(f"Structure breakeven(s): {strategy['breakevens']}; "
                     f"max loss {strategy.get('max_loss')}, max profit {strategy.get('max_profit')}.")
    if result.get("costs"):
        lines.append(f"Estimated round-trip cost: "
                     f"Rs.{result['costs']['estimated_round_trip']:,.0f}.")
    if result.get("event_risk", {}).get("level") in ("HIGH", "ELEVATED"):
        lines.append(f"Event risk is {result['event_risk']['level']}: "
                     f"{result['event_risk']['flags'][0]}")
    for note in result.get("structure_reasoning", []):
        lines.append(note)

    # A playbook pointing the other way is worth saying out loud rather than
    # burying: it is the clearest available evidence that the setup is contested.
    signal_dir = result["signal"]["direction"]
    conflicting = [p for p in playbooks
                   if p.direction.value not in (signal_dir, "NEUTRAL")]
    if conflicting:
        top = conflicting[0]
        lines.append(
            f"CONFLICT: the '{top.name}' playbook points {top.direction.value} while the "
            f"signal is {signal_dir}. {top.reason} Size down or wait for the conflict to "
            f"resolve."
        )

    return {
        "action": action, "summary": summary, "details": lines,
        "confidence": round(signal.confidence, 1),
        "playbooks": [p.to_dict() for p in playbooks[:3]],
        "conflicting_playbooks": [p.to_dict() for p in conflicting],
    }


DISCLAIMER = (
    "This is analysis software, not investment advice, and it is not a prediction. "
    "It applies published technical and options methods to price data and shows its "
    "reasoning; it has no knowledge of news, order flow or anything else that actually "
    "moves the market. No configuration of it will produce reliably profitable trading. "
    "Backtest it on your own data, paper trade it for months, and risk only capital you "
    "can afford to lose. Derivatives can lose more than you put in."
)

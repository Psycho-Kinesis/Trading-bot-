"""Event-driven backtester.

Design constraints, and what they cost you:

**No look-ahead.**  At bar ``i`` the engine rebuilds the entire context from
``ohlcv[:i+1]`` only, and entries fill at bar ``i+1``'s open.  This is slower
than vectorising, but vectorised indicator frames leak future information
through any centred or forward-shifted calculation, and that leak is exactly
what produces backtests that cannot be reproduced live.

**Synthetic option pricing.**  Historical Indian option chains are not freely
available, so in ``mode="options"`` the engine *prices options with
Black-Scholes* from the index path and an assumed implied volatility.  This is
an approximation with real consequences:

* It has no volatility smile, so OTM options are priced too cheaply.
* It has no IV path, so it misses the IV crush that follows events -- the very
  thing that most often turns a correct directional call into a losing trade.
* It has no bid-ask, so the spread is modelled as a flat percentage.

Results from options mode are therefore *optimistic* about option buying.  Use
``mode="index"`` (P&L in index points, as a futures proxy) when you want a
clean read on whether the *signal* has an edge, and treat options-mode numbers
as an upper bound rather than an expectation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..constants import IST, Direction
from ..indicators._util import ensure_ohlcv
from ..options.pricing import bs_price, greeks, time_to_expiry
from ..options.selection import select_expiry
from ..risk.costs import CostModel
from ..risk.manager import RiskLimits, RiskManager
from ..risk.sizing import size_option_position
from ..signals.engine import EngineConfig, SignalEngine, build_context
from .metrics import PerformanceReport, compute_metrics

__all__ = ["BacktestConfig", "Backtester", "Trade"]


@dataclass
class BacktestConfig:
    initial_capital: float = 500_000.0
    mode: str = "options"                # "options" | "index"
    symbol: str = "NIFTY"
    timeframe: str = "1d"

    warmup_bars: int = 250               # bars of history each context sees
    stride: int = 1                      # evaluate every Nth bar
    max_hold_bars: int = 5               # time stop
    bars_per_day: int = 1                # 1 for daily bars; 75 for 5-minute bars
    roll_expiry_for_hold: bool = True    # buy an expiry that outlives the hold
    allow_pyramiding: bool = False

    assumed_iv: float = 0.14             # used when no IV series is supplied
    target_delta: float = 0.42
    risk_pct: float = 0.01
    max_premium_pct: float = 0.25
    premium_stop_pct: float = 0.40       # cut a long option at -40%

    include_costs: bool = True
    include_slippage: bool = True
    exchange: str = "NSE"
    lot_size: int | None = None          # None = look up from the calendar

    engine_config: EngineConfig = field(default_factory=EngineConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class Trade:
    entry_time: object
    exit_time: object = None
    direction: str = ""
    action: str = ""
    entry_spot: float = 0.0
    exit_spot: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    lots: int = 0
    units: int = 0
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    strike: float | None = None
    gross_pnl: float = 0.0
    costs: float = 0.0
    pnl: float = 0.0
    risk: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0
    bars_held: int = 0
    regime: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Backtester:
    """Walk a price series bar by bar, generating and simulating signals."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.engine = SignalEngine(self.config.engine_config)
        self.costs = CostModel(exchange=self.config.exchange,
                               include_slippage=self.config.include_slippage)
        self._skips: dict[str, int] = {}
        self._capital_shortfalls: list[float] = []

    def _note_skip(self, reason: str) -> None:
        """Record why a signal was not taken, bucketed so the tally stays readable."""
        self._skips[reason] = self._skips.get(reason, 0) + 1

    def _note_capital_shortfall(self, risk_per_lot: float, equity: float) -> None:
        self._capital_shortfalls.append(risk_per_lot)
        self._note_skip("capital too small for one lot at the configured risk limit")

    # -- option helpers ----------------------------------------------------
    def _iv_at(self, index, iv_series: pd.Series | None) -> float:
        if iv_series is None:
            return self.config.assumed_iv
        try:
            value = float(iv_series.loc[:index].iloc[-1])
            return value / 100 if value > 1.5 else value
        except (KeyError, IndexError, ValueError):
            return self.config.assumed_iv

    def _pick_strike(self, spot: float, t: float, iv: float, option_type: str,
                     step: int) -> float:
        from ..options.pricing import delta_to_strike
        return delta_to_strike(self.config.target_delta, spot, t, iv, option_type, step)

    # -- main loop ---------------------------------------------------------
    def run(
        self, ohlcv: pd.DataFrame, higher_tf: pd.DataFrame | None = None,
        iv_series: pd.Series | None = None, vix_series: pd.Series | None = None,
        progress: bool = False,
    ) -> "BacktestResult":
        cfg = self.config
        d = ensure_ohlcv(ohlcv)
        if not isinstance(d.index, pd.DatetimeIndex):
            raise ValueError("backtest needs a DatetimeIndex on the OHLCV frame")
        d = d.sort_index()
        n = len(d)
        if n < cfg.warmup_bars + 10:
            raise ValueError(f"need at least {cfg.warmup_bars + 10} bars, got {n}")

        self._skips, self._capital_shortfalls = {}, []
        risk = RiskManager(capital=cfg.initial_capital, limits=cfg.risk_limits)
        equity = cfg.initial_capital
        equity_curve: list[tuple[object, float]] = []
        trades: list[Trade] = []
        open_trade: Trade | None = None
        open_state: dict = {}
        skipped: dict[str, int] = {}

        opens = d["open"].to_numpy(float)
        highs = d["high"].to_numpy(float)
        lows = d["low"].to_numpy(float)
        closes = d["close"].to_numpy(float)

        for i in range(cfg.warmup_bars, n - 1):
            ts = d.index[i]
            day = ts.date() if hasattr(ts, "date") else None
            if day:
                risk.roll_day(day)

            # --- manage an open position on this bar --------------------
            if open_trade is not None:
                exit_reason, exit_spot = self._check_exit(
                    open_trade, open_state, highs[i], lows[i], closes[i], i, ts
                )
                if exit_reason:
                    self._close_trade(open_trade, open_state, exit_spot, ts, exit_reason,
                                      i, iv_series)
                    equity += open_trade.pnl
                    risk.record_trade(open_trade.pnl)
                    risk.open_positions.clear()
                    risk.deployed_capital = 0.0
                    trades.append(open_trade)
                    open_trade, open_state = None, {}

            equity_curve.append((ts, equity))

            if open_trade is not None and not cfg.allow_pyramiding:
                continue
            if (i - cfg.warmup_bars) % cfg.stride != 0:
                continue

            # --- evaluate a new signal ----------------------------------
            window = d.iloc[max(0, i - cfg.warmup_bars + 1): i + 1]
            htf_window = None
            if higher_tf is not None:
                htf_window = higher_tf.loc[:ts]
                if len(htf_window) < 60:
                    htf_window = None

            vix = None
            if vix_series is not None:
                try:
                    vix = float(vix_series.loc[:ts].iloc[-1])
                except (KeyError, IndexError, ValueError):
                    vix = None

            now = ts.to_pydatetime()
            if now.tzinfo is None:
                now = now.replace(tzinfo=IST)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctx = build_context(cfg.symbol, window, cfg.timeframe,
                                    higher_tf_ohlcv=htf_window, india_vix=vix, now=now)
                signal = self.engine.generate(ctx)

            if not signal.is_actionable:
                skipped_key = _bucket_veto(signal.vetoes)
                skipped[skipped_key] = skipped.get(skipped_key, 0) + 1
                continue

            # --- size and open at the NEXT bar's open -------------------
            entry_spot = float(opens[i + 1])
            entry_ts = d.index[i + 1]
            trade = self._open_trade(signal, entry_spot, entry_ts, ctx, risk, equity,
                                     iv_series, i + 1)
            if trade is None:
                continue
            open_trade, open_state = trade
            risk.open_positions.append({"direction": open_trade.direction})
            risk.deployed_capital += open_trade.entry_premium * open_trade.units

        # Force-close anything still open at the end of the data.
        if open_trade is not None:
            self._close_trade(open_trade, open_state, float(closes[-1]), d.index[-1],
                              "END_OF_DATA", n - 1, iv_series)
            equity += open_trade.pnl
            trades.append(open_trade)
        equity_curve.append((d.index[-1], equity))

        trade_frame = pd.DataFrame([t.to_dict() for t in trades])
        eq = pd.Series(dict(equity_curve)).sort_index()
        report = compute_metrics(trade_frame, eq, cfg.initial_capital)
        for key, count in self._skips.items():
            skipped[key] = skipped.get(key, 0) + count

        diagnostics: list[str] = []
        if self._capital_shortfalls:
            typical = float(np.median(self._capital_shortfalls))
            needed = typical / cfg.risk_pct
            diagnostics.append(
                f"{len(self._capital_shortfalls)} signal(s) were skipped because one lot "
                f"risked more than the {cfg.risk_pct:.0%} per-trade limit -- typically "
                f"Rs.{typical:,.0f} per lot. Trading this instrument on this timeframe at "
                f"that risk limit needs roughly Rs.{needed:,.0f} of capital. Either raise "
                f"capital, trade options (where risk per lot is the premium, not the index "
                f"move), or use a shorter timeframe with tighter stops."
            )
        if not len(trade_frame):
            diagnostics.append(
                "No trades were taken at all. That is a result, not a failure: check the "
                "skip tally below to see whether the filters are too strict or the data "
                "genuinely offered no setups."
            )
        return BacktestResult(report=report, trades=trade_frame, equity=eq,
                              config=cfg, skipped=skipped, diagnostics=diagnostics)

    # -- trade lifecycle ---------------------------------------------------
    def _open_trade(self, signal, entry_spot, entry_ts, ctx, risk, equity,
                    iv_series, bar_index):
        cfg = self.config
        bullish = signal.direction == Direction.UP
        option_type = "CE" if bullish else "PE"

        exp = ctx.expiry or {}
        lot = cfg.lot_size or exp.get("lot_size") or 75
        step = exp.get("strike_step", 50)

        # Buy an expiry that outlives the intended hold.  Without this the
        # position expires inside the holding window and every OTM option
        # settles at zero regardless of whether the signal was right -- which
        # is what a naive "always trade the nearest weekly" backtest does, and
        # why its results are meaningless.
        hold_days = max(1, cfg.max_hold_bars // max(cfg.bars_per_day, 1))
        dte = max(exp.get("dte_trading", 3) or 3, 1)
        if cfg.roll_expiry_for_hold and cfg.mode == "options":
            try:
                chosen = select_expiry(cfg.symbol, entry_ts.date(),
                                       intended_hold_days=hold_days)
                dte = max(chosen["dte_trading"], 1)
            except (KeyError, ValueError, AttributeError):
                dte = max(dte, hold_days + 1)
        t = time_to_expiry(dte)
        iv = self._iv_at(entry_ts, iv_series)

        # Shift the risk geometry to the actual fill.
        drift = entry_spot - signal.spot
        stop = signal.stop_loss + drift
        target = signal.targets[0] + drift
        stop_distance = abs(entry_spot - stop)
        if stop_distance <= 0:
            return None

        if cfg.mode == "index":
            from ..risk.sizing import fixed_fractional

            lots = fixed_fractional(equity, cfg.risk_pct, stop_distance, lot)
            if lots < 1:
                # Worth surfacing rather than silently skipping: one NIFTY lot
                # with a 250-point stop risks ~Rs.19,000, which is 3.8% of a
                # Rs.5 lakh account. Small accounts simply cannot take index
                # futures at a 1% risk budget.
                self._note_capital_shortfall(stop_distance * lot, equity)
                return None
            units = lots * lot
            sizing_risk = stop_distance * units
            check = risk.check(signal.direction.value, sizing_risk, 0.0)
            if not check.allowed:
                self._note_skip(check.reasons[0][:60] if check.reasons else "risk limit")
                return None
            trade = Trade(
                entry_time=entry_ts, direction=signal.direction.value,
                action=signal.action.value, entry_spot=entry_spot, stop_loss=stop,
                target=target, lots=lots, units=units, entry_premium=entry_spot,
                risk=sizing_risk, confidence=signal.confidence,
                regime=(signal.regime or {}).get("regime", ""),
            )
            return trade, {"bar_index": bar_index, "mode": "index", "iv": iv,
                           "dte": dte, "lot": lot}

        strike = self._pick_strike(entry_spot, t, iv, option_type, step)
        premium = bs_price(entry_spot, strike, t, iv, option_type)
        if premium <= 0.5:
            return None
        g = greeks(entry_spot, strike, t, iv, option_type)

        sizing = size_option_position(
            capital=equity, risk_pct=cfg.risk_pct, premium=premium, lot_size=lot,
            delta=g.delta, index_stop_distance=stop_distance,
            max_premium_pct=cfg.max_premium_pct,
            stop_is_premium_pct=cfg.premium_stop_pct,
        )
        if sizing.lots < 1:
            self._note_skip(sizing.reasons[-1][:70] if sizing.reasons else "sizing -> 0 lots")
            return None

        check = risk.check(signal.direction.value, sizing.risk_amount,
                           premium * sizing.units,
                           is_expiry_day=bool(exp.get("is_expiry_day")))
        if not check.allowed:
            self._note_skip(check.reasons[0][:70] if check.reasons else "risk limit")
            return None

        trade = Trade(
            entry_time=entry_ts, direction=signal.direction.value,
            action=signal.action.value, entry_spot=entry_spot, stop_loss=stop,
            target=target, lots=sizing.lots, units=sizing.units,
            entry_premium=premium, strike=strike, risk=sizing.risk_amount,
            confidence=signal.confidence, regime=(signal.regime or {}).get("regime", ""),
        )
        return trade, {
            "bar_index": bar_index, "mode": "options", "iv": iv, "dte": dte,
            "lot": lot, "option_type": option_type, "strike": strike,
            "entry_bar": bar_index, "moneyness": _moneyness_bucket(entry_spot, strike, option_type),
            "is_expiry_day": bool(exp.get("is_expiry_day")),
        }

    def _check_exit(self, trade, state, high, low, close, bar_index, ts):
        """Stop and target are evaluated on the index path.

        When both the stop and the target are touched inside the same bar the
        engine assumes the *stop* filled first.  Without intrabar data there is
        no way to know the order, and assuming the favourable one is how
        backtests quietly inflate their win rate.
        """
        bullish = trade.direction == "UP"
        held = bar_index - state["bar_index"]

        if bullish:
            if low <= trade.stop_loss:
                return "STOP", trade.stop_loss
            if high >= trade.target:
                return "TARGET", trade.target
        else:
            if high >= trade.stop_loss:
                return "STOP", trade.stop_loss
            if low <= trade.target:
                return "TARGET", trade.target

        if held >= self.config.max_hold_bars:
            return "TIME_STOP", close
        if state.get("mode") == "options":
            days_held = held / max(self.config.bars_per_day, 1)
            if days_held >= state.get("dte", 99):
                return "EXPIRY", close
        return None, close

    def _close_trade(self, trade, state, exit_spot, exit_ts, reason, bar_index, iv_series):
        cfg = self.config
        trade.exit_time = exit_ts
        trade.exit_spot = exit_spot
        trade.exit_reason = reason
        trade.bars_held = bar_index - state["bar_index"]

        if state.get("mode") == "index":
            sign = 1 if trade.direction == "UP" else -1
            trade.gross_pnl = sign * (exit_spot - trade.entry_spot) * trade.units
            trade.exit_premium = exit_spot
            cost = self.costs.futures_round_trip(trade.entry_spot, exit_spot, trade.units) \
                if cfg.include_costs else None
            trade.costs = cost.total if cost else 0.0
        else:
            days_held = trade.bars_held / max(cfg.bars_per_day, 1)
            remaining_dte = max(state["dte"] - days_held, 0)
            t = time_to_expiry(remaining_dte) if remaining_dte > 0 else 0.0
            iv = self._iv_at(exit_ts, iv_series)
            exit_premium = bs_price(exit_spot, state["strike"], t, iv, state["option_type"])
            trade.exit_premium = exit_premium
            trade.gross_pnl = (exit_premium - trade.entry_premium) * trade.units

            # Honour the premium stop: a long option is usually cut on premium
            # loss well before the index stop is reached.
            max_loss = -trade.entry_premium * cfg.premium_stop_pct * trade.units
            if trade.gross_pnl < max_loss and reason not in ("EXPIRY", "END_OF_DATA"):
                trade.gross_pnl = max_loss
                trade.exit_reason = f"{reason}+PREMIUM_STOP"

            cost = self.costs.option_round_trip(
                trade.entry_premium, max(exit_premium, 0.05), trade.units, "LONG",
                state.get("moneyness", "ATM"), state.get("is_expiry_day", False),
            ) if cfg.include_costs else None
            trade.costs = cost.total if cost else 0.0

        trade.pnl = trade.gross_pnl - trade.costs


def _bucket_veto(vetoes: list[str]) -> str:
    """Collapse near-identical veto strings into one reportable category."""
    if not vetoes:
        return "no directional edge (score inside the neutral band)"
    first = vetoes[0]
    for needle, label in (
        ("Reward/risk", "reward/risk below minimum"),
        ("room to the next", "entry too close to a level (wall)"),
        ("Higher timeframe", "counter to the higher timeframe"),
        ("volatile chop", "regime is volatile chop"),
        ("compressed", "volatility squeeze with no direction"),
        ("Confidence", "confidence below threshold"),
        ("bars of history", "insufficient history"),
    ):
        if needle in first:
            return label
    return first[:60]


def _moneyness_bucket(spot: float, strike: float, option_type: str) -> str:
    diff = abs(strike - spot) / spot
    if diff < 0.003:
        return "ATM"
    if diff < 0.015:
        return "OTM"
    return "FAR_OTM"


@dataclass
class BacktestResult:
    report: PerformanceReport
    trades: pd.DataFrame
    equity: pd.Series
    config: BacktestConfig
    skipped: dict
    diagnostics: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 68,
                 f"BACKTEST  {self.config.symbol}  {self.config.timeframe}  "
                 f"mode={self.config.mode}",
                 "=" * 68]
        lines += self.report.summary_lines()
        if self.config.mode == "options":
            lines += ["", "NOTE: option prices are synthetic (Black-Scholes, flat IV of "
                          f"{self.config.assumed_iv:.0%}). No smile, no IV path, no real "
                          "spreads -- these results are optimistic for option buying."]
        caveats = self.report.caveats()
        if caveats:
            lines += ["", "CAVEATS:"] + [f"  - {c}" for c in caveats]
        if self.diagnostics:
            lines += ["", "DIAGNOSTICS:"] + [f"  - {d}" for d in self.diagnostics]
        if self.skipped:
            lines += ["", "Signals not taken (top reasons):"]
            for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1])[:5]:
                lines.append(f"  {count:5}x  {reason}")
        return "\n".join(lines)

"""Named trading setups.

The rule engine produces a *directional opinion*.  A playbook is a specific,
named trade with its own entry trigger, invalidation and regime gate --
the difference between "the market looks bullish" and "this is an opening
range breakout, long above 24,910, out below 24,860".

Each playbook declares:

* the regimes it is valid in (and is skipped outside them),
* the session phases it applies to,
* an explicit trigger, invalidation and reason,
* a ``fails_when`` string naming its known failure mode, because a setup you
  cannot describe the failure of is one you will keep taking after it stops
  working.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import Direction, Regime
from .base import MarketContext

__all__ = ["PLAYBOOKS", "Playbook", "PlaybookMatch", "match_playbooks"]


@dataclass
class PlaybookMatch:
    name: str
    direction: Direction
    trigger: float | None
    invalidation: float | None
    confidence: float          # 0..1, the playbook's own conviction
    reason: str
    fails_when: str
    horizon: str

    def to_dict(self) -> dict:
        return {
            "name": self.name, "direction": self.direction.value,
            "trigger": round(self.trigger, 2) if self.trigger else None,
            "invalidation": round(self.invalidation, 2) if self.invalidation else None,
            "confidence": round(self.confidence, 2), "reason": self.reason,
            "fails_when": self.fails_when, "horizon": self.horizon,
        }


@dataclass
class Playbook:
    name: str
    regimes: tuple[Regime, ...]
    phases: tuple[str, ...]
    horizon: str
    fails_when: str
    detector: object = None
    timeframes: tuple[str, ...] = ()

    def applies(self, ctx: MarketContext) -> bool:
        if self.regimes and ctx.regime.regime not in self.regimes:
            return False
        if self.phases and ctx.session_phase not in self.phases and ctx.timeframe != "1d":
            return False
        return not (self.timeframes and ctx.timeframe not in self.timeframes)


# ---------------------------------------------------------------------------
# Detectors.  Each returns a PlaybookMatch or None.
# ---------------------------------------------------------------------------

def _trend_pullback(ctx: MarketContext) -> PlaybookMatch | None:
    """Buy the pullback to a rising 20 EMA inside an established uptrend.

    The highest-expectancy setup in most trend-following literature, and the
    one that requires the most patience -- the point is to enter *after* the
    retracement, not into strength.
    """
    close = ctx.get("close")
    ema20, ema50 = ctx.get("ema_20"), ctx.get("ema_50")
    rsi = ctx.get("rsi_14")
    atr = ctx.atr
    if None in (close, ema20, ema50, rsi) or atr <= 0:
        return None

    bullish = ctx.regime.is_bullish and ema20 > ema50
    bearish = ctx.regime.is_bearish and ema20 < ema50
    if not (bullish or bearish):
        return None

    distance = abs(close - ema20) / atr
    if distance > 1.0:
        return None                        # not a pullback, price is extended

    if bullish and 35 <= rsi <= 60:
        swing_low = getattr(ctx.structure, "swing_low", None)
        return PlaybookMatch(
            "trend_pullback_long", Direction.UP,
            trigger=close + 0.25 * atr,
            invalidation=(swing_low if swing_low else ema50) - 0.3 * atr,
            confidence=0.62 + 0.15 * ctx.regime.trend_strength,
            reason=(f"Uptrend intact (20EMA above 50EMA) and price has pulled back to "
                    f"within {distance:.2f} ATR of the 20 EMA with RSI at {rsi:.0f} -- "
                    f"a retracement inside a trend, not a reversal."),
            fails_when=("The pullback becomes a reversal. If price closes below the "
                        "50 EMA or takes out the last swing low, the trend structure "
                        "is broken and this setup is void, not 'more oversold'."),
            horizon="2-5 sessions",
        )
    if bearish and 40 <= rsi <= 65:
        swing_high = getattr(ctx.structure, "swing_high", None)
        return PlaybookMatch(
            "trend_pullback_short", Direction.DOWN,
            trigger=close - 0.25 * atr,
            invalidation=(swing_high if swing_high else ema50) + 0.3 * atr,
            confidence=0.62 + 0.15 * ctx.regime.trend_strength,
            reason=(f"Downtrend intact (20EMA below 50EMA) and price has bounced to "
                    f"within {distance:.2f} ATR of the 20 EMA with RSI at {rsi:.0f}."),
            fails_when=("A close above the 50 EMA or a break of the last swing high "
                        "invalidates the downtrend structure."),
            horizon="2-5 sessions",
        )
    return None


def _squeeze_breakout(ctx: MarketContext) -> PlaybookMatch | None:
    """Trade the expansion out of a volatility squeeze.

    The squeeze itself is directionless; this only fires once the expansion has
    actually started, because compressed markets break both ways and often
    fake one first.
    """
    if not ctx.get("squeeze_fired"):
        return None
    momentum = ctx.get("squeeze_momentum")
    # squeeze_bars is already 0 on the firing bar; the prior count is the
    # length of the compression that just resolved.
    bars = ctx.get("squeeze_bars_prior") or ctx.get("squeeze_bars") or 0
    if momentum is None:
        return None
    atr = ctx.atr
    close = ctx.get("close")
    if close is None or atr <= 0:
        return None

    up = momentum > 0
    dc_upper, dc_lower = ctx.get("dc_upper"), ctx.get("dc_lower")
    return PlaybookMatch(
        "squeeze_breakout_long" if up else "squeeze_breakout_short",
        Direction.UP if up else Direction.DOWN,
        trigger=(dc_upper if up and dc_upper else close + 0.3 * atr) if up
                else (dc_lower if dc_lower else close - 0.3 * atr),
        invalidation=close - 1.2 * atr if up else close + 1.2 * atr,
        confidence=0.55 + min(0.2, bars * 0.02),
        reason=(f"Volatility compressed for {int(bars)} bars and has just started to "
                f"expand {'upward' if up else 'downward'}. Range expansion out of "
                f"compression is the cleanest context for buying option premium, "
                f"because it is the one time IV is cheap and about to rise."),
        fails_when=("The break fails and price returns inside the range. A squeeze "
                    "that fires and reverses within two bars is a trap -- exit rather "
                    "than reverse, since the second break often fails too."),
        horizon="1-4 sessions",
    )


def _liquidity_sweep_reversal(ctx: MarketContext) -> PlaybookMatch | None:
    """Fade a stop-hunt: price wicks through a swing and closes back inside."""
    close, atr = ctx.get("close"), ctx.atr
    if close is None or atr <= 0:
        return None
    # A sweep only matters if price is still near the swept level. A sweep two
    # bars ago at a level 20 ATR away is history, and pairing today's price with
    # that level would produce a nonsensical stop.
    sweeps = [
        s for s in (ctx.snapshot.get("_sweeps") or [])
        if s.get("bars_ago", 99) <= 2 and abs(close - s.get("level", close)) <= atr * 2.0
    ]
    if not sweeps:
        return None
    sweep = sweeps[0]
    up = sweep["bias"] > 0
    return PlaybookMatch(
        "sweep_reversal_long" if up else "sweep_reversal_short",
        Direction.UP if up else Direction.DOWN,
        trigger=close + 0.2 * atr if up else close - 0.2 * atr,
        invalidation=sweep["level"] - 0.2 * atr if up else sweep["level"] + 0.2 * atr,
        confidence=0.55 + min(0.2, sweep.get("wick_atr", 0.5) * 0.15),
        reason=(f"{sweep['kind'].replace('_', ' ').lower()} at {sweep['level']:.0f}: "
                f"{sweep['note']}. The wick shows the level was tested and rejected, "
                f"which is a far better entry than chasing the break."),
        fails_when=("Price closes back through the swept level -- then it was a real "
                    "break, not a sweep, and the level has flipped."),
        horizon="intraday to 2 sessions",
    )


def _range_fade(ctx: MarketContext) -> PlaybookMatch | None:
    """Fade the edges of an established range.

    Only valid in a RANGE regime with a real level nearby.  Fading a range
    edge in a trend is how accounts die.
    """
    info = ctx.snapshot.get("_veto_levels_info") or {}
    close, atr = ctx.get("close"), ctx.atr
    if close is None or atr <= 0 or not info.get("at_level"):
        return None
    bb_pct = ctx.get("bb_pct_b")
    rsi = ctx.get("rsi_14")
    if bb_pct is None or rsi is None:
        return None

    support = info.get("nearest_support")
    resistance = info.get("nearest_resistance")
    room_up = info.get("room_to_resistance_atr")
    room_down = info.get("room_to_support_atr")

    if bb_pct < 0.12 and rsi < 38 and support and room_up and room_up > 1.5:
        return PlaybookMatch(
            "range_fade_long", Direction.UP,
            trigger=close + 0.15 * atr, invalidation=support - 0.5 * atr,
            confidence=0.52,
            reason=(f"Price is at the lower band and at support {support:.0f} inside a "
                    f"range, with RSI at {rsi:.0f} and {room_up:.1f} ATR of room back "
                    f"to resistance."),
            fails_when=("The range breaks. A close below the support level means this "
                        "was a breakdown, and range-fade entries have no stop discipline "
                        "built in unless you enforce one."),
            horizon="1-3 sessions",
        )
    if bb_pct > 0.88 and rsi > 62 and resistance and room_down and room_down > 1.5:
        return PlaybookMatch(
            "range_fade_short", Direction.DOWN,
            trigger=close - 0.15 * atr, invalidation=resistance + 0.5 * atr,
            confidence=0.52,
            reason=(f"Price is at the upper band and at resistance {resistance:.0f} "
                    f"inside a range, RSI {rsi:.0f}, with {room_down:.1f} ATR of room "
                    f"back to support."),
            fails_when=("The range breaks upward. Fading resistance in what turns out "
                        "to be a breakout is the classic way to be short a trend."),
            horizon="1-3 sessions",
        )
    return None


def _structure_break_retest(ctx: MarketContext) -> PlaybookMatch | None:
    """Enter on the retest after a break of structure, not on the break itself."""
    st = ctx.structure
    if st is None or not st.last_event or st.bars_since_event is None:
        return None
    if not (1 <= st.bars_since_event <= 6):
        return None
    close, atr = ctx.get("close"), ctx.atr
    if close is None or atr <= 0:
        return None

    up = st.last_event.endswith("_UP")
    level = st.swing_high if up else st.swing_low
    if level is None:
        return None
    # A retest means price has come back to the broken level.
    if abs(close - level) > atr * 1.2:
        return None

    return PlaybookMatch(
        "bos_retest_long" if up else "bos_retest_short",
        Direction.UP if up else Direction.DOWN,
        trigger=close + 0.2 * atr if up else close - 0.2 * atr,
        invalidation=level - 0.8 * atr if up else level + 0.8 * atr,
        confidence=0.6,
        reason=(f"{st.last_event.replace('_', ' ').title()} {st.bars_since_event} bars "
                f"ago and price has returned to retest the broken level at "
                f"{level:.0f}. Entering on the retest gives a much tighter "
                f"invalidation than chasing the break did."),
        fails_when=("Price closes back through the level in the original direction -- "
                    "then the break failed and the move is a liquidity grab."),
        horizon="2-5 sessions",
    )


def _expiry_theta_burn(ctx: MarketContext) -> PlaybookMatch | None:
    """Not a directional trade -- an explicit warning on expiry day."""
    exp = ctx.expiry or {}
    if not exp.get("is_expiry_day"):
        return None
    return PlaybookMatch(
        "expiry_day_caution", Direction.NEUTRAL, trigger=None, invalidation=None,
        confidence=0.0,
        reason=("Expiry day. Premium decays toward intrinsic through the session and "
                "gamma is extreme after about 14:00. Long options usually go to zero; "
                "short options can lose several times the credit collected on one "
                "fast move. Directional index views are better expressed in the next "
                "expiry."),
        fails_when="This is a caution, not a setup -- there is nothing here to enter.",
        horizon="today only",
    )


PLAYBOOKS: list[Playbook] = [
    Playbook("trend_pullback",
             (Regime.STRONG_UPTREND, Regime.UPTREND, Regime.DOWNTREND, Regime.STRONG_DOWNTREND),
             (), "2-5 sessions",
             "the pullback turns into a trend reversal", _trend_pullback),
    Playbook("squeeze_breakout",
             (Regime.SQUEEZE, Regime.RANGE, Regime.UPTREND, Regime.DOWNTREND),
             (), "1-4 sessions",
             "the breakout fails back inside the range", _squeeze_breakout),
    Playbook("liquidity_sweep_reversal", (), (), "intraday to 2 sessions",
             "the sweep was a genuine break", _liquidity_sweep_reversal),
    Playbook("range_fade", (Regime.RANGE,), (), "1-3 sessions",
             "the range breaks instead of holding", _range_fade),
    Playbook("bos_retest",
             (Regime.STRONG_UPTREND, Regime.UPTREND, Regime.DOWNTREND,
              Regime.STRONG_DOWNTREND, Regime.RANGE),
             (), "2-5 sessions",
             "the break was false and price reclaims the level", _structure_break_retest),
    Playbook("expiry_caution", (), (), "today only",
             "not applicable -- this is a warning", _expiry_theta_burn),
]


def match_playbooks(ctx: MarketContext) -> list[PlaybookMatch]:
    """Every playbook whose regime gate passes and whose detector fires."""
    out: list[PlaybookMatch] = []
    for playbook in PLAYBOOKS:
        if not playbook.applies(ctx):
            continue
        try:
            match = playbook.detector(ctx)
        except Exception:
            continue
        if match is not None:
            out.append(match)
    return sorted(out, key=lambda m: -m.confidence)

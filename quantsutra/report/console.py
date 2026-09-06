"""Terminal rendering for signals, regimes and backtests."""

from __future__ import annotations

import datetime as dt

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..constants import Action, Direction

__all__ = ["render_signal", "render_regime", "render_chain", "render_backtest",
           "render_events", "console"]

console = Console()

DIRECTION_STYLE = {Direction.UP: "bold green", Direction.DOWN: "bold red",
                   Direction.NEUTRAL: "bold yellow"}
ACTION_STYLE = {
    Action.BUY_CALL: "bold green", Action.BUY_PUT: "bold red",
    Action.NO_TRADE: "bold yellow", Action.IRON_CONDOR: "cyan",
    Action.LONG_STRADDLE: "cyan", Action.SHORT_STRADDLE: "magenta",
}


def _score_bar(value: float, width: int = 21) -> Text:
    """A centred bar for a score in [-1, 1]."""
    mid = width // 2
    filled = int(abs(value) * mid)
    bar = [" "] * width
    bar[mid] = "|"
    if value > 0:
        for i in range(mid + 1, min(width, mid + 1 + filled)):
            bar[i] = "#"
    elif value < 0:
        for i in range(max(0, mid - filled), mid):
            bar[i] = "#"
    text = Text("".join(bar))
    text.stylize("green" if value > 0 else "red" if value < 0 else "dim")
    return text


def render_regime(regime: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("Regime", Text(regime["regime"], style="bold"))
    table.add_row("Trend score", f"{regime['trend_score']:+.2f}")
    table.add_row("Trend strength", f"{regime['trend_strength']:.2f}")
    table.add_row("Volatility", regime["volatility_state"])
    table.add_row("ADX / Choppiness", f"{regime['adx']:.0f} / {regime['choppiness']:.0f}")
    table.add_row("ATR percentile", f"{regime['atr_percentile']:.0f}")
    items = [table]
    if regime.get("notes"):
        items.append(Text(""))
        for note in regime["notes"]:
            items.append(Text(f"  ! {note}", style="yellow"))
    return Panel(Group(*items), title="Market regime", border_style="blue")


def render_signal(signal: dict, show_rules: bool = True, max_rules: int = 12) -> Group:
    """Render a signal dict (from ``Signal.to_dict()``)."""
    direction = Direction(signal["direction"])
    action = Action(signal["action"])
    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim", justify="right")
    header.add_column()
    header.add_row("Symbol", Text(f"{signal['symbol']}  ({signal['timeframe']})", style="bold"))
    header.add_row("Spot", f"{signal['spot']:,.2f}")
    header.add_row("Direction", Text(direction.value, style=DIRECTION_STYLE.get(direction, "")))
    header.add_row("Action", Text(action.value, style=ACTION_STYLE.get(action, "bold")))
    header.add_row("Confidence", _confidence_text(signal["confidence"]))
    header.add_row("Net score", _score_bar(signal["raw_score"]))

    blocks = [Panel(header, title="Signal", border_style="cyan")]

    if signal.get("entry") and signal.get("stop_loss"):
        risk = abs(signal["entry"] - signal["stop_loss"])
        trade = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        trade.add_column("Level"); trade.add_column("Price", justify="right")
        trade.add_column("Distance", justify="right")
        trade.add_row("Entry", f"{signal['entry']:,.2f}", "-")
        trade.add_row("Stop loss", f"{signal['stop_loss']:,.2f}", f"{risk:,.0f} pts (1R)")
        for i, target in enumerate(signal.get("targets", []), 1):
            r = abs(target - signal["entry"]) / risk if risk else 0
            trade.add_row(f"Target {i}", f"{target:,.2f}", f"{r:.1f}R")
        if signal.get("risk_reward"):
            trade.add_row("Reward/risk", f"{signal['risk_reward']:.2f}", "to T1")
        blocks.append(Panel(trade, title="Trade plan (index points)", border_style="green"))

    if signal.get("contract"):
        blocks.append(Panel(_contract_table(signal["contract"]),
                            title="Option contract", border_style="magenta"))
    if signal.get("position"):
        blocks.append(Panel(_position_table(signal["position"]),
                            title="Position size", border_style="magenta"))

    if signal.get("reasons"):
        reasons = Text()
        for line in signal["reasons"]:
            style = "green" if line.startswith("[^]") else "red" if line.startswith("[v]") else "white"
            reasons.append(f"  {line}\n", style=style)
        blocks.append(Panel(reasons, title="Why", border_style="white"))

    if signal.get("category_scores") and show_rules:
        cat = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        cat.add_column("Family"); cat.add_column("Score", justify="right"); cat.add_column("")
        for name, value in sorted(signal["category_scores"].items(), key=lambda kv: -abs(kv[1])):
            cat.add_row(name, f"{value:+.2f}", _score_bar(value, 17))
        blocks.append(Panel(cat, title="Evidence by family", border_style="blue"))

    if show_rules and signal.get("rules"):
        rules = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
        rules.add_column("Rule", style="dim"); rules.add_column("Bias")
        rules.add_column("Score", justify="right"); rules.add_column("Rationale", overflow="fold")
        fired = [r for r in signal["rules"] if abs(r["score"]) > 0.01 or r["confidence"] < 1]
        for rule in sorted(fired, key=lambda r: -abs(r["score"]))[:max_rules]:
            style = "green" if rule["score"] > 0 else "red" if rule["score"] < 0 else "dim"
            rules.add_row(rule["name"], Text(rule["bias"], style=style),
                          f"{rule['score']:+.2f}", rule["rationale"])
        blocks.append(Panel(rules, title="Rules that fired", border_style="blue"))

    if signal.get("vetoes"):
        veto = Text()
        for line in signal["vetoes"]:
            veto.append(f"  X {line}\n", style="bold red")
        blocks.append(Panel(veto, title="Vetoes -- this trade is blocked", border_style="red"))

    if signal.get("warnings"):
        warn = Text()
        for line in signal["warnings"]:
            warn.append(f"  ! {line}\n", style="yellow")
        blocks.append(Panel(warn, title="Warnings", border_style="yellow"))

    return Group(*blocks)


def _confidence_text(value: float) -> Text:
    style = "bold green" if value >= 72 else "yellow" if value >= 55 else "red"
    bar = "#" * int(value / 5) + "." * (20 - int(value / 5))
    return Text(f"{value:5.1f}  {bar}", style=style)


def _contract_table(contract: dict) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right"); table.add_column()
    for key in ("instrument", "expiry", "dte_trading", "strike", "option_type",
                "premium", "delta", "iv", "oi", "moneyness", "lot_size"):
        if contract.get(key) is not None:
            table.add_row(key.replace("_", " ").title(), str(contract[key]))
    if contract.get("reasons"):
        table.add_row("", "")
        for reason in contract["reasons"]:
            table.add_row("", Text(reason, style="dim"))
    return table


def _position_table(position: dict) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right"); table.add_column()
    for key in ("lots", "units", "risk_amount", "risk_pct_of_capital", "max_loss",
                "notional", "margin_estimate"):
        if position.get(key) is not None:
            value = position[key]
            table.add_row(key.replace("_", " ").title(),
                          f"{value:,.2f}" if isinstance(value, float) else str(value))
    for reason in position.get("reasons", []):
        table.add_row("", Text(reason, style="dim"))
    for warning in position.get("warnings", []):
        table.add_row("", Text(f"! {warning}", style="yellow"))
    return table


def render_chain(summary: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right"); table.add_column()
    for key in ("symbol", "spot", "expiry", "dte_trading", "atm_strike", "atm_iv",
                "pcr_oi", "pcr_reading", "iv_skew", "skew_reading",
                "max_call_oi_strike", "max_put_oi_strike", "oi_flow_bias",
                "max_pain", "max_pain_distance_pct"):
        if summary.get(key) is not None:
            table.add_row(key.replace("_", " ").title(), str(summary[key]))
    items = [table]
    for warning in summary.get("warnings", []):
        items.append(Text(f"  ! {warning}", style="yellow"))
    return Panel(Group(*items), title="Option chain", border_style="magenta")


def render_events(risk: dict) -> Panel:
    style = {"HIGH": "red", "ELEVATED": "yellow"}.get(risk["level"], "green")
    body = Text()
    body.append(f"  Event risk: {risk['level']}\n\n", style=f"bold {style}")
    for flag in risk["flags"]:
        body.append(f"  - {flag}\n", style=style if risk["level"] != "NORMAL" else "white")
    for event in risk.get("upcoming", []):
        body.append(f"  > {event['event']} in {event['days_away']} day(s) "
                    f"({event['iv_impact']} IV impact)\n", style="cyan")
    return Panel(body, title="Event risk", border_style=style)


def render_backtest(result) -> Panel:
    body = Text()
    for line in result.report.summary_lines():
        body.append(f"  {line}\n")
    if result.config.mode == "options":
        body.append("\n  NOTE: option prices are synthetic (Black-Scholes, flat IV). "
                    "No smile, no IV path, no real spreads -- optimistic for buying.\n",
                    style="yellow")
    caveats = result.report.caveats()
    if caveats:
        body.append("\n  CAVEATS\n", style="bold yellow")
        for caveat in caveats:
            body.append(f"  - {caveat}\n", style="yellow")
    for diagnostic in getattr(result, "diagnostics", []):
        body.append(f"\n  {diagnostic}\n", style="cyan")
    if result.skipped:
        body.append("\n  Signals not taken\n", style="bold dim")
        for reason, count in sorted(result.skipped.items(), key=lambda kv: -kv[1])[:6]:
            body.append(f"  {count:5}x  {reason}\n", style="dim")
    return Panel(body, title="Backtest", border_style="blue")

"""Command-line interface.

    quantsutra signal NIFTY            analyse and print a trade recommendation
    quantsutra chain NIFTY             option chain analytics
    quantsutra regime NIFTY            regime and volatility state only
    quantsutra backtest NIFTY          walk-forward simulation with costs
    quantsutra calendar                expiries, holidays and event risk
    quantsutra costs --premium 200     what a trade actually costs
    quantsutra events                  historical episodes and their lessons
    quantsutra demo                    full pipeline on synthetic data, offline
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import warnings

import click

from . import __version__
from .constants import IST

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# The calendar's provisional/missing-holiday warnings are surfaced properly in
# the rendered output, so the raw stderr copy is noise here.
warnings.filterwarnings("ignore", message=".*[Hh]oliday list.*")
warnings.filterwarnings("ignore", message=".*No holiday list configured.*")

SYMBOLS = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]


def _console():
    from .report.console import console
    return console


def _load_history(symbol, source, interval, lookback, directory):
    """Fetch price history from the chosen source, with a clear failure path."""
    from .data import CsvFeed, FeedError, YahooFeed, generate_index_series

    if source == "synthetic":
        return generate_index_series(days=lookback, seed=7), []
    feed = CsvFeed(directory) if source == "csv" else YahooFeed()
    try:
        frame = feed.history(symbol, interval, lookback)
    except FeedError as exc:
        raise click.ClickException(
            f"{exc}\n\nTry: --source csv --dir <folder> with an exported file, or "
            f"--source synthetic to see the tool run offline."
        ) from exc
    from .data.base import validate_frame
    return frame, validate_frame(frame, symbol)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="quantsutra")
def cli():
    """Analysis engine for Indian index derivatives (NIFTY, BANKNIFTY, SENSEX).

    Not investment advice. See `quantsutra disclaimer`.
    """


@cli.command()
@click.argument("symbol", default="NIFTY")
@click.option("--source", type=click.Choice(["yahoo", "csv", "synthetic"]), default="yahoo",
              help="Where price history comes from.")
@click.option("--dir", "directory", default="data", help="Directory for --source csv.")
@click.option("--interval", default="1d", help="Bar interval (1d, 15m, 5m).")
@click.option("--lookback", default=500, help="Bars of history to load.")
@click.option("--capital", default=500_000.0, help="Account capital in rupees.")
@click.option("--risk", "risk_pct", default=1.0, help="Risk per trade, in percent.")
@click.option("--chain/--no-chain", default=False, help="Fetch the live NSE option chain.")
@click.option("--delta", "target_delta", default=0.42, help="Target option delta.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a report.")
@click.option("--rules/--no-rules", default=True, help="Show the individual rules.")
def signal(symbol, source, directory, interval, lookback, capital, risk_pct,
           chain, target_delta, as_json, rules):
    """Analyse SYMBOL and print a complete trade recommendation."""
    from .analysis import AnalysisConfig, analyze
    from .data.base import resample_ohlcv

    console = _console()
    frame, issues = _load_history(symbol, source, interval, lookback, directory)

    higher = None
    if interval == "1d" and len(frame) > 120:
        higher = resample_ohlcv(frame, "1W")

    option_chain, vix, vix_history = None, None, None
    if chain:
        from .data import FeedError, NseFeed
        try:
            nse = NseFeed()
            option_chain = nse.option_chain(symbol)
            vix = nse.india_vix()
        except (FeedError, Exception) as exc:
            console.print(f"[yellow]Could not fetch the option chain: {exc}[/yellow]")
            console.print("[dim]Continuing without it -- direction and levels are still "
                          "computed, but no strike will be selected.[/dim]\n")

    if source == "yahoo":
        from .data import FeedError, YahooFeed
        try:
            vix_history = YahooFeed().india_vix(250)
            vix = vix if vix is not None else float(vix_history.iloc[-1])
        except (FeedError, Exception):
            pass

    result = analyze(
        symbol, frame, interval, higher_tf=higher, chain=option_chain,
        india_vix=vix, vix_history=vix_history,
        config=AnalysisConfig(capital=capital, risk_pct=risk_pct / 100,
                              target_delta=target_delta),
    )

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    _print_analysis(console, result, issues, show_rules=rules)


def _print_analysis(console, result, issues, show_rules=True):
    from rich.panel import Panel
    from rich.text import Text

    from .report.console import (render_events, render_regime, render_signal)

    if issues:
        body = Text()
        for issue in issues:
            body.append(f"  ! {issue}\n", style="yellow")
        console.print(Panel(body, title="Data quality", border_style="yellow"))

    if result.get("error"):
        console.print(Panel(Text(result["error"], style="red"), title="Cannot analyse",
                            border_style="red"))
        return

    console.print(render_signal(result["signal"], show_rules=show_rules))
    console.print(render_regime(result["signal"]["regime"]))

    rec = result.get("recommendation") or {}
    if rec:
        body = Text()
        body.append(f"  {rec.get('summary', '')}\n\n", style="bold")
        for line in rec.get("details", []):
            style = "red" if line.startswith("CONFLICT") else "white"
            body.append(f"  - {line}\n", style=style)
        console.print(Panel(body, title="Recommendation", border_style="green"))

    if result.get("playbooks"):
        from rich.table import Table
        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        table.add_column("Playbook"); table.add_column("Dir"); table.add_column("Conf", justify="right")
        table.add_column("Trigger", justify="right"); table.add_column("Invalidation", justify="right")
        table.add_column("Fails when", overflow="fold")
        for pb in result["playbooks"]:
            table.add_row(pb["name"], pb["direction"], f"{pb['confidence']:.2f}",
                          str(pb["trigger"] or "-"), str(pb["invalidation"] or "-"),
                          pb["fails_when"])
        console.print(Panel(table, title="Playbooks", border_style="cyan"))

    levels = result.get("levels") or {}
    if levels:
        from rich.table import Table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right"); table.add_column()
        table.add_row("ATR", str(levels.get("atr")))
        table.add_row("Nearest support", str(levels.get("nearest_support")))
        table.add_row("Nearest resistance", str(levels.get("nearest_resistance")))
        if levels.get("confluence_zones"):
            table.add_row("Confluence", ", ".join(
                f"{z['price']:.0f} ({'+'.join(z['sources'])})"
                for z in levels["confluence_zones"][:3]))
        for gap in (levels.get("fair_value_gaps") or [])[:2]:
            table.add_row("Unfilled FVG", f"{gap['bottom']:.0f}-{gap['top']:.0f} "
                                          f"({gap['kind'].lower()}, {gap['bars_ago']} bars ago)")
        console.print(Panel(table, title="Levels", border_style="blue"))

    if result.get("chain"):
        from .report.console import render_chain
        console.print(render_chain(result["chain"]))

    if result.get("event_risk"):
        console.print(render_events(result["event_risk"]))

    if result.get("historical_lessons"):
        body = Text()
        for lesson in result["historical_lessons"]:
            body.append(f"  - {lesson}\n", style="dim")
        console.print(Panel(body, title="What history says about this regime",
                            border_style="dim"))

    console.print(Panel(Text(result["disclaimer"], style="dim italic"),
                        border_style="red", title="Read this"))


@cli.command()
@click.argument("symbol", default="NIFTY")
@click.option("--expiry", default=None, help="Expiry as YYYY-MM-DD (default: nearest).")
@click.option("--strikes", default=6, help="Strikes to show either side of ATM.")
@click.option("--json", "as_json", is_flag=True)
def chain(symbol, expiry, strikes, as_json):
    """Fetch and analyse the live option chain for SYMBOL."""
    from .data import FeedError, NseFeed
    from .report.console import render_chain

    console = _console()
    target = dt.date.fromisoformat(expiry) if expiry else None
    try:
        option_chain = NseFeed().option_chain(symbol, target)
    except (FeedError, Exception) as exc:
        raise click.ClickException(
            f"Could not fetch the NSE option chain: {exc}\n\n"
            f"NSE blocks datacentre IPs and rate-limits aggressively. This works from a "
            f"home connection; on a cloud host it usually will not."
        ) from exc

    option_chain.compute_ivs()
    summary = option_chain.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2, default=str))
        return

    console.print(render_chain(summary))
    from rich.panel import Panel
    table = option_chain.greeks_table(strikes)
    if not table.empty:
        from rich.table import Table
        rich_table = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
        for column in ("strike", "type", "moneyness", "price", "iv", "delta", "gamma",
                       "theta", "vega"):
            rich_table.add_column(column.title(), justify="right")
        for _, row in table.iterrows():
            rich_table.add_row(f"{row['strike']:.0f}", row["type"], row["moneyness"],
                               f"{row['price']:.2f}", f"{row['iv']:.1f}",
                               f"{row['delta']:+.3f}", f"{row['gamma']:.5f}",
                               f"{row['theta']:.2f}", f"{row['vega']:.2f}")
        console.print(Panel(rich_table, title="Greeks around ATM", border_style="magenta"))


@cli.command()
@click.argument("symbol", default="NIFTY")
@click.option("--source", type=click.Choice(["yahoo", "csv", "synthetic"]), default="yahoo")
@click.option("--dir", "directory", default="data")
@click.option("--interval", default="1d")
@click.option("--lookback", default=500)
def regime(symbol, source, directory, interval, lookback):
    """Print the current market regime and volatility state for SYMBOL."""
    from .regime import classify_regime
    from .report.console import render_regime

    console = _console()
    frame, issues = _load_history(symbol, source, interval, lookback, directory)
    for issue in issues:
        console.print(f"[yellow]! {issue}[/yellow]")
    state = classify_regime(frame)
    console.print(render_regime(state.to_dict()))

    from .knowledge.events import lessons_for_regime
    lessons = lessons_for_regime(state.regime.value, state.volatility_state)
    if lessons:
        from rich.panel import Panel
        from rich.text import Text
        body = Text()
        for lesson in lessons:
            body.append(f"  - {lesson}\n", style="dim")
        console.print(Panel(body, title="Historical context", border_style="dim"))


@cli.command()
@click.argument("symbol", default="NIFTY")
@click.option("--source", type=click.Choice(["yahoo", "csv", "synthetic"]), default="synthetic")
@click.option("--dir", "directory", default="data")
@click.option("--lookback", default=900, help="Bars of history to simulate over.")
@click.option("--capital", default=1_000_000.0)
@click.option("--risk", "risk_pct", default=1.0, help="Risk per trade, percent.")
@click.option("--mode", type=click.Choice(["options", "index"]), default="options")
@click.option("--hold", "max_hold", default=5, help="Max bars to hold a position.")
@click.option("--warmup", default=250, help="Bars of context each signal sees.")
@click.option("--costs/--no-costs", default=True)
@click.option("--save", default=None, help="Write the trade log to this CSV path.")
def backtest(symbol, source, directory, lookback, capital, risk_pct, mode, max_hold,
             warmup, costs, save):
    """Bar-by-bar backtest with realistic Indian transaction costs.

    For out-of-sample validation use `quantsutra walkforward` instead -- a
    single backtest over one history proves very little on its own.
    """
    from .backtest import BacktestConfig, Backtester
    from .report.console import render_backtest

    console = _console()
    frame, issues = _load_history(symbol, source, interval="1d",
                                  lookback=lookback, directory=directory)
    for issue in issues:
        console.print(f"[yellow]! {issue}[/yellow]")

    if len(frame) < warmup + 30:
        raise click.ClickException(
            f"Need at least {warmup + 30} bars for a backtest with --warmup {warmup}; "
            f"got {len(frame)}. Increase --lookback or reduce --warmup."
        )

    console.print(f"[dim]Simulating {len(frame) - warmup} bars... "
                  f"(the engine rebuilds full context per bar, so this is not instant)[/dim]")
    config = BacktestConfig(symbol=symbol, mode=mode, initial_capital=capital,
                            risk_pct=risk_pct / 100, max_hold_bars=max_hold,
                            warmup_bars=warmup, include_costs=costs)
    result = Backtester(config).run(frame)
    console.print(render_backtest(result))

    if save and len(result.trades):
        result.trades.to_csv(save, index=False)
        console.print(f"[green]Trade log written to {save}[/green]")


@cli.command("walkforward")
@click.argument("symbol", default="NIFTY")
@click.option("--source", type=click.Choice(["yahoo", "csv", "synthetic"]), default="synthetic")
@click.option("--dir", "directory", default="data")
@click.option("--lookback", default=1600, help="Total bars of history to split.")
@click.option("--train", "train_bars", default=400, help="In-sample bars per fold.")
@click.option("--test", "test_bars", default=300, help="Out-of-sample bars per fold.")
@click.option("--warmup", default=250)
@click.option("--capital", default=1_000_000.0)
@click.option("--mode", type=click.Choice(["options", "index"]), default="options")
def walkforward(symbol, source, directory, lookback, train_bars, test_bars, warmup,
                capital, mode):
    """Out-of-sample validation: does the behaviour survive unseen data?

    Reports the efficiency ratio -- out-of-sample expectancy divided by
    in-sample expectancy. A ratio near 1 means it generalised; near 0 means the
    in-sample result was fitting.
    """
    from rich.panel import Panel
    from rich.text import Text

    from .backtest import BacktestConfig, walk_forward

    console = _console()
    frame, issues = _load_history(symbol, source, "1d", lookback, directory)
    for issue in issues:
        console.print(f"[yellow]! {issue}[/yellow]")

    console.print(f"[dim]Running walk-forward over {len(frame)} bars "
                  f"({train_bars} train / {test_bars} test per fold). This runs a full "
                  f"backtest per fold, so it takes a while.[/dim]")
    config = BacktestConfig(symbol=symbol, mode=mode, initial_capital=capital,
                            warmup_bars=warmup)
    try:
        result = walk_forward(frame, config, train_bars=train_bars, test_bars=test_bars,
                              progress=True)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not result.folds:
        raise click.ClickException(
            "No folds completed. Increase --lookback or reduce --train/--test.")

    efficiency = result.efficiency()
    style = {"GENERALISED": "green", "FAILED_OUT_OF_SAMPLE": "red",
             "LIKELY_OVERFIT": "red", "INCONSISTENT": "yellow"}.get(
        efficiency["verdict"], "yellow")
    console.print(Panel(Text(result.summary()), title="Walk-forward",
                        border_style=style))


@cli.command()
@click.argument("symbol", default="NIFTY")
@click.option("--days", default=30, help="Days ahead to list.")
def calendar(symbol, days):
    """Expiries, holidays and event risk for the coming period."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from .calendar_in import (expiry_chain, holiday_data_status, holidays,
                              is_trading_day, lot_size, next_trading_day)
    from .knowledge.events import event_risk
    from .report.console import render_events

    console = _console()
    today = dt.date.today()

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("Expiry"); table.add_column("Weekday"); table.add_column("Days away", justify="right")
    for expiry in expiry_chain(symbol, today, 5):
        table.add_row(expiry.isoformat(), expiry.strftime("%A"), str((expiry - today).days))
    console.print(Panel(table, title=f"{symbol} expiries (lot size {lot_size(symbol)})",
                        border_style="cyan"))

    upcoming = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    upcoming.add_column("Date"); upcoming.add_column("Holiday")
    found = False
    for date, name in sorted(holidays().items()):
        if today <= date <= today + dt.timedelta(days=days):
            upcoming.add_row(date.isoformat(), name)
            found = True
    if found:
        console.print(Panel(upcoming, title=f"Holidays in the next {days} days",
                            border_style="blue"))

    status = holiday_data_status(today.year)
    if status != "confirmed":
        console.print(Panel(
            Text(f"Holiday data for {today.year} is {status}. Expiry dates near a holiday "
                 f"may be wrong. Update config/holidays.yaml from the exchange circular.",
                 style="yellow"),
            border_style="yellow", title="Calendar warning"))

    console.print(render_events(event_risk(today, symbol)))
    if not is_trading_day(today):
        console.print(f"[yellow]Market is closed today. Next trading day: "
                      f"{next_trading_day(today)}[/yellow]")


@cli.command()
@click.option("--premium", default=200.0, help="Option premium per unit.")
@click.option("--lots", default=1, help="Number of lots.")
@click.option("--symbol", default="NIFTY")
@click.option("--moneyness", type=click.Choice(["ATM", "OTM", "FAR_OTM"]), default="ATM")
@click.option("--expiry-day", is_flag=True, help="Apply wider expiry-day spreads.")
def costs(premium, lots, symbol, moneyness, expiry_day):
    """What a trade actually costs: STT, GST, stamp duty, spread."""
    from rich.panel import Panel
    from rich.table import Table

    from .calendar_in import lot_size
    from .risk.costs import CostModel

    console = _console()
    lot = lot_size(symbol)
    quantity = lot * lots
    model = CostModel(exchange="BSE" if symbol.upper() in ("SENSEX", "BANKEX") else "NSE")
    breakeven = model.breakeven_move(premium, quantity, moneyness, expiry_day)

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("Component"); table.add_column("Rupees", justify="right")
    for key, value in breakeven["breakdown"].items():
        if key in ("statutory", "total"):
            continue
        table.add_row(key.replace("_", " ").title(), f"{value:,.2f}")
    table.add_row("", "")
    table.add_row("Statutory + brokerage", f"{breakeven['breakdown']['statutory']:,.2f}")
    table.add_row("Assumed spread cost", f"{breakeven['breakdown']['slippage']:,.2f}")
    table.add_row("TOTAL round trip", f"{breakeven['total_cost']:,.2f}")
    console.print(Panel(
        table,
        title=f"{lots} lot(s) of {symbol} ({quantity} units) at premium {premium}",
        border_style="magenta"))

    console.print(Panel(
        f"  The premium must move [bold]{breakeven['premium_move_pct']:.2f}%[/bold] "
        f"(Rs.{breakeven['premium_move_needed']:.2f} per unit) just to break even.\n\n"
        f"  [dim]The spread assumption dominates and is a default, not a measurement. "
        f"Check your own bid-ask before trusting this.[/dim]",
        title="Breakeven", border_style="yellow"))


@cli.command()
@click.option("--category", default=None, help="Filter by category (CRASH, ELECTION, ...).")
def events(category):
    """Historical Indian market episodes and what each one teaches."""
    from rich.panel import Panel
    from rich.text import Text

    from .knowledge.events import MARKET_EVENTS, RECURRING_EVENTS

    console = _console()
    body = Text()
    for event in MARKET_EVENTS:
        if category and event.category.upper() != category.upper():
            continue
        move = f"{event.approx_move_pct:+.0f}%" if event.approx_move_pct is not None else "-"
        body.append(f"  {event.date}  {event.name}  [{event.category} {move}]\n",
                    style="bold")
        body.append(f"    {event.description}\n", style="dim")
        body.append(f"    Lesson: {event.lesson}\n\n", style="cyan")
    console.print(Panel(body, title="Historical episodes", border_style="blue"))

    recurring = Text()
    for item in RECURRING_EVENTS:
        recurring.append(f"  {item['name']} ({item['when']}) -- IV impact "
                         f"{item['iv_impact']}, typical move {item['typical_move_pct']}%\n",
                         style="bold")
        recurring.append(f"    {item['note']}\n\n", style="dim")
    console.print(Panel(recurring, title="Recurring calendar events", border_style="cyan"))


@cli.command()
@click.option("--capital", default=1_000_000.0)
def demo(capital):
    """Run the whole pipeline on synthetic data -- no network needed."""
    import numpy as np

    from .analysis import AnalysisConfig, analyze
    from .data import generate_index_series, generate_vix_series
    from .data.base import resample_ohlcv
    from .options.chain import chain_from_records
    from .options.pricing import bs_price, time_to_expiry

    console = _console()
    console.print("[bold cyan]Running on generated data -- none of this is real "
                  "market data.[/bold cyan]\n")

    frame = generate_index_series(days=600, seed=42)
    weekly = resample_ohlcv(frame, "1W")
    vix_history = generate_vix_series(frame)

    spot = float(frame["close"].iloc[-1])
    step, dte = 50, 4
    t = time_to_expiry(dte)
    base = round(spot / step) * step
    records = []
    for strike in np.arange(base - 1500, base + 1550, step):
        strike = float(strike)
        m = (strike - spot) / spot
        iv = 0.13 + 0.9 * m**2 - 0.55 * m
        oi = int(400_000 * np.exp(-((abs(m) * 100) ** 2) / 8)) + 30_000
        records += [
            {"strike": strike, "type": "CE", "ltp": round(bs_price(spot, strike, t, iv, "CE"), 2),
             "oi": oi, "iv": round(iv * 100, 2), "volume": oi // 3, "oi_change": 8000},
            {"strike": strike, "type": "PE", "ltp": round(bs_price(spot, strike, t, iv, "PE"), 2),
             "oi": int(oi * 1.15), "iv": round(iv * 100, 2), "volume": oi // 2, "oi_change": 12000},
        ]
    option_chain = chain_from_records("NIFTY", spot, dt.date.today() + dt.timedelta(days=6),
                                      records, dte_trading=dte, lot_size=75, strike_step=step)

    result = analyze("NIFTY", frame, "1d", higher_tf=weekly, chain=option_chain,
                     india_vix=float(vix_history.iloc[-1]), vix_history=vix_history,
                     config=AnalysisConfig(capital=capital))
    _print_analysis(console, result, result.get("data_quality", []))


@cli.command()
@click.argument("symbols", nargs=-1)
@click.option("--interval", default=300, help="Seconds between scans.")
@click.option("--timeframe", default="15m", help="Bar interval to analyse.")
@click.option("--capital", default=500_000.0)
@click.option("--risk", "risk_pct", default=1.0)
@click.option("--chain/--no-chain", default=False, help="Fetch the NSE option chain.")
@click.option("--telegram", is_flag=True,
              help="Also send to Telegram (needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).")
@click.option("--journal", "journal_path", default="quantsutra_journal.sqlite")
@click.option("--once", is_flag=True, help="Scan once and exit.")
def watch(symbols, interval, timeframe, capital, risk_pct, chain, telegram,
          journal_path, once):
    """Scan SYMBOLS on a loop during market hours and alert on changes.

    Places no orders. Every signal, actionable or not, is written to the
    journal so the record can be audited later.
    """
    from .analysis import AnalysisConfig
    from .live import ConsoleSink, LiveScanner, Notifier, ScannerConfig, TelegramSink

    console = _console()
    symbols = tuple(s.upper() for s in symbols) or ("NIFTY",)

    sinks = [ConsoleSink()]
    if telegram:
        sink = TelegramSink()
        if not sink.configured:
            raise click.ClickException(
                "Telegram needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the "
                "environment. They are read from there deliberately so credentials "
                "never land in a config file.")
        sinks.append(sink)

    config = ScannerConfig(
        symbols=symbols, interval_seconds=interval, timeframe=timeframe,
        fetch_chain=chain, journal_path=journal_path,
        analysis=AnalysisConfig(capital=capital, risk_pct=risk_pct / 100),
        max_iterations=1 if once else None,
    )
    scanner = LiveScanner(config, notifier=Notifier(sinks=sinks))
    console.print(f"[cyan]Watching {', '.join(symbols)} on {timeframe} every "
                  f"{interval}s. Analysis only -- no orders are placed.[/cyan]")
    try:
        if once:
            for result in scanner.scan_once():
                if "error" in result:
                    console.print(f"[red]{result['symbol']}: {result['error']}[/red]")
                else:
                    _print_analysis(console, result, result.get("data_quality", []),
                                    show_rules=False)
        else:
            scanner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    finally:
        scanner.close()


@cli.command("journal")
@click.option("--path", "journal_path", default="quantsutra_journal.sqlite")
@click.option("--symbol", default=None)
@click.option("--limit", default=15)
def journal_cmd(journal_path, symbol, limit):
    """Audit the signal log: hit rate, confidence interval, results by regime."""
    from pathlib import Path

    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from .live import Journal

    console = _console()
    if not Path(journal_path).exists():
        raise click.ClickException(
            f"No journal at {journal_path}. Run `quantsutra watch` to start one.")

    with Journal(journal_path) as journal:
        stats = journal.hit_rate(symbol)
        body = Text()
        body.append(f"  Actionable signals : {stats['actionable_signals']}\n")
        body.append(f"  Resolved           : {stats['resolved']}\n")
        body.append(f"  Unresolved         : {stats['unresolved']}\n")
        if stats.get("hit_rate") is not None:
            low, high = stats["hit_rate_ci"]
            body.append(f"  Hit rate           : {stats['hit_rate']:.1%} "
                        f"(95% CI {low:.0%}-{high:.0%})\n")
            if stats.get("avg_r") is not None:
                body.append(f"  Average R          : {stats['avg_r']:+.2f}\n")
        if stats.get("note"):
            body.append(f"\n  {stats['note']}\n", style="yellow")
        console.print(Panel(body, title="Realised record", border_style="cyan"))

        by_regime = journal.stats_by_regime()
        if by_regime:
            table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
            table.add_column("Regime"); table.add_column("Signals", justify="right")
            table.add_column("Hit rate", justify="right"); table.add_column("Avg R", justify="right")
            for row in by_regime:
                table.add_row(str(row["regime"]), str(row["signals"]),
                              f"{row['hit_rate']:.0%}" if row["hit_rate"] is not None else "-",
                              f"{row['avg_r']:+.2f}" if row["avg_r"] is not None else "-")
            console.print(Panel(table, title="By regime", border_style="blue"))

        recent = journal.recent(symbol, limit)
        if recent:
            table = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
            for column in ("id", "created_at", "symbol", "action", "conf", "spot", "regime"):
                table.add_column(column)
            for row in recent:
                table.add_row(str(row["id"]), str(row["created_at"])[:16], row["symbol"],
                              row["action"] or "-",
                              f"{row['confidence']:.0f}" if row["confidence"] else "-",
                              f"{row['spot']:.0f}" if row["spot"] else "-",
                              row["regime"] or "-")
            console.print(Panel(table, title="Recent signals", border_style="dim"))


@cli.command()
def disclaimer():
    """What this software is and is not."""
    from rich.panel import Panel

    from .analysis import DISCLAIMER

    _console().print(Panel(DISCLAIMER, title="Disclaimer", border_style="red"))


def main():  # pragma: no cover
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()

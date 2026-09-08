"""Backtester: no look-ahead, honest fills, correct accounting."""

import numpy as np
import pandas as pd
import pytest

from quantsutra.backtest import BacktestConfig, Backtester, compute_metrics
from quantsutra.backtest.metrics import win_rate_confidence_interval


def test_wilson_interval_widens_with_a_smaller_sample():
    small = win_rate_confidence_interval(18, 30)
    large = win_rate_confidence_interval(600, 1000)
    assert (small[1] - small[0]) > (large[1] - large[0])
    assert small[0] < 0.5 < small[1], "a 30-trade sample cannot exclude 'no edge'"


def test_metrics_on_an_empty_trade_log():
    report = compute_metrics(pd.DataFrame(), pd.Series(dtype=float), 500_000)
    assert report.metrics["trades"] == 0
    assert report.metrics["net_pnl"] == 0


def test_metrics_flag_a_small_sample():
    trades = pd.DataFrame({"pnl": [100.0] * 20, "costs": [10.0] * 20})
    equity = pd.Series(500_000 + np.cumsum(trades["pnl"]))
    caveats = compute_metrics(trades, equity, 500_000).caveats()
    assert any("cannot distinguish an edge from luck" in c for c in caveats)


def test_metrics_flag_a_strategy_that_only_works_gross():
    rng = np.random.default_rng(0)
    gross = rng.normal(80, 700, 150)
    costs = np.full(150, 180.0)
    trades = pd.DataFrame({"pnl": gross - costs, "costs": costs})
    equity = pd.Series(500_000 + np.cumsum(trades["pnl"]))
    report = compute_metrics(trades, equity, 500_000)
    if report.metrics["gross_pnl"] > 0 >= report.metrics["net_pnl"]:
        assert any("unprofitable after" in c for c in report.caveats())


def test_metrics_flag_dependence_on_one_trade():
    trades = pd.DataFrame({"pnl": [-100.0] * 20 + [50_000.0], "costs": [10.0] * 21})
    equity = pd.Series(500_000 + np.cumsum(trades["pnl"]))
    caveats = compute_metrics(trades, equity, 500_000).caveats()
    assert any("best trade" in c for c in caveats)


def test_sharpe_does_not_penalise_cash_days():
    """A profitable but mostly-flat equity curve must not report a large
    negative Sharpe just because it held cash."""
    equity = pd.Series([500_000] * 200)
    equity.iloc[100:] = 520_000
    trades = pd.DataFrame({"pnl": [20_000.0], "costs": [200.0], "bars_held": [3]})
    metrics = compute_metrics(trades, equity, 500_000).metrics
    assert metrics["sharpe"] >= 0


def _series(n=700, seed=11):
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(n // 3, 0.0009), np.full(n // 3, -0.0006),
                            np.full(n - 2 * (n // 3), 0.0002)])
    close = 20000 * np.exp(np.cumsum(drift + rng.normal(0, 0.0085, n)))
    frame = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "close": close, "volume": rng.integers(2e5, 9e5, n),
    }, index=pd.bdate_range("2022-01-03", periods=n))
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


@pytest.mark.slow
def test_backtest_runs_and_accounts_correctly():
    frame = _series()
    result = Backtester(BacktestConfig(mode="options", warmup_bars=250, max_hold_bars=6,
                                       initial_capital=1_500_000)).run(frame)
    assert result.equity.iloc[-1] == pytest.approx(
        1_500_000 + result.trades["pnl"].sum() if len(result.trades) else 1_500_000, rel=1e-6
    )
    if len(result.trades):
        trades = result.trades
        assert (trades["pnl"] == trades["gross_pnl"] - trades["costs"]).all()
        assert (trades["costs"] >= 0).all()
        assert (trades["exit_time"] >= trades["entry_time"]).all()
        assert (trades["lots"] >= 1).all()


@pytest.mark.slow
def test_backtest_reports_why_signals_were_skipped():
    result = Backtester(BacktestConfig(mode="options", warmup_bars=250,
                                       initial_capital=1_500_000)).run(_series())
    assert result.skipped, "the engine must account for the bars it did not trade"
    assert sum(result.skipped.values()) > 0


@pytest.mark.slow
def test_backtest_requires_enough_history():
    with pytest.raises(ValueError, match="at least"):
        Backtester(BacktestConfig(warmup_bars=250)).run(_series(n=100))


@pytest.mark.slow
def test_costs_reduce_net_pnl():
    frame = _series()
    config_with = BacktestConfig(mode="options", warmup_bars=250, max_hold_bars=6,
                                 initial_capital=1_500_000, include_costs=True)
    config_without = BacktestConfig(mode="options", warmup_bars=250, max_hold_bars=6,
                                    initial_capital=1_500_000, include_costs=False)
    with_costs = Backtester(config_with).run(frame)
    without_costs = Backtester(config_without).run(frame)
    if len(with_costs.trades) and len(without_costs.trades):
        assert with_costs.trades["costs"].sum() > 0
        assert without_costs.trades["costs"].sum() == 0
        assert with_costs.report.metrics["net_pnl"] < without_costs.report.metrics["net_pnl"]


@pytest.mark.slow
def test_options_mode_holds_an_expiry_that_outlives_the_hold():
    """Without rolling the expiry, every OTM option settles at zero regardless
    of whether the signal was right."""
    result = Backtester(BacktestConfig(mode="options", warmup_bars=250, max_hold_bars=6,
                                       initial_capital=1_500_000)).run(_series())
    if len(result.trades):
        expiry_exits = result.trades["exit_reason"].str.contains("EXPIRY").mean()
        assert expiry_exits < 0.5, "most trades should not be dying at expiry"


@pytest.mark.slow
def test_backtest_summary_warns_about_synthetic_option_prices():
    result = Backtester(BacktestConfig(mode="options", warmup_bars=250,
                                       initial_capital=1_500_000)).run(_series())
    assert "synthetic" in result.summary().lower()


# --- buy-and-hold benchmark ----------------------------------------------

def test_buy_and_hold_computes_return_and_drawdown():
    from quantsutra.backtest import buy_and_hold

    prices = pd.Series([100.0, 120.0, 80.0, 110.0],
                       index=pd.bdate_range("2024-01-01", periods=4))
    result = buy_and_hold(prices, 1_000_000)
    assert result["benchmark_return"] == pytest.approx(0.10)
    assert result["benchmark_pnl"] == pytest.approx(100_000)
    # Worst point is 80 against a running peak of 120.
    assert result["benchmark_max_drawdown"] == pytest.approx(-1 / 3, rel=1e-3)


def test_buy_and_hold_handles_degenerate_input():
    from quantsutra.backtest import buy_and_hold

    assert buy_and_hold(pd.Series([100.0]), 1_000) == {}
    assert buy_and_hold(pd.Series(dtype=float), 1_000) == {}


def test_metrics_flag_underperforming_buy_and_hold():
    """Trading 50 times to finish behind a single buy order is the result that
    matters, and it must not be buried."""
    trades = pd.DataFrame({"pnl": [1000.0] * 50, "costs": [100.0] * 50})
    equity = pd.Series(1_000_000 + np.cumsum(trades["pnl"]),
                       index=pd.bdate_range("2024-01-01", periods=50))
    # Strategy makes 5%; the index doubles.
    prices = pd.Series(np.linspace(100, 200, 50),
                       index=pd.bdate_range("2024-01-01", periods=50))
    report = compute_metrics(trades, equity, 1_000_000, benchmark_prices=prices)
    assert report.metrics["benchmark_return"] == pytest.approx(1.0)
    assert any("Underperformed buy-and-hold" in c for c in report.caveats())
    assert any("Buy & hold" in line for line in report.summary_lines())


def test_metrics_do_not_flag_when_the_strategy_wins():
    trades = pd.DataFrame({"pnl": [20_000.0] * 50, "costs": [100.0] * 50})
    equity = pd.Series(1_000_000 + np.cumsum(trades["pnl"]),
                       index=pd.bdate_range("2024-01-01", periods=50))
    prices = pd.Series(np.linspace(100, 102, 50),
                       index=pd.bdate_range("2024-01-01", periods=50))
    report = compute_metrics(trades, equity, 1_000_000, benchmark_prices=prices)
    assert not any("Underperformed buy-and-hold" in c for c in report.caveats())


def test_benchmark_is_absent_when_no_prices_are_supplied():
    trades = pd.DataFrame({"pnl": [100.0] * 10, "costs": [10.0] * 10})
    equity = pd.Series(1_000_000 + np.cumsum(trades["pnl"]))
    metrics = compute_metrics(trades, equity, 1_000_000).metrics
    assert metrics.get("benchmark_return") is None


@pytest.mark.slow
def test_backtest_reports_the_benchmark():
    result = Backtester(BacktestConfig(mode="options", warmup_bars=250,
                                       initial_capital=1_500_000)).run(_series())
    assert result.report.metrics.get("benchmark_return") is not None
    assert "Buy & hold" in result.summary()

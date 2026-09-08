"""Walk-forward validation and the paper broker."""

from itertools import pairwise

import pytest

from quantsutra.backtest import BacktestConfig, WalkForwardResult, rolling_windows, walk_forward
from quantsutra.brokers import Order, OrderType, PaperBroker


def test_rolling_windows_are_consecutive_and_non_overlapping_on_test():
    windows = list(rolling_windows(1000, 400, 150))
    assert windows[0] == (slice(0, 400), slice(400, 550))
    test_slices = [t for _, t in windows]
    for earlier, later in pairwise(test_slices):
        assert later.start >= earlier.stop, "test windows must not overlap"


def test_walk_forward_rejects_folds_shorter_than_the_warmup(synthetic):
    config = BacktestConfig(warmup_bars=250)
    with pytest.raises(ValueError, match="exceed warmup_bars"):
        walk_forward(synthetic, config, train_bars=200, test_bars=200)


def test_walk_forward_rejects_insufficient_history(synthetic):
    config = BacktestConfig(warmup_bars=100)
    with pytest.raises(ValueError, match="at least"):
        walk_forward(synthetic.head(200), config, train_bars=150, test_bars=150)


def test_efficiency_reports_insufficient_data_rather_than_a_number():
    result = WalkForwardResult(folds=[{
        "fold": 1, "is": {"trades": 1, "win_rate": 1.0, "expectancy_r": 2.0,
                          "net_pnl": 1, "max_drawdown": 0, "profit_factor": None},
        "oos": {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0,
                "net_pnl": 0, "max_drawdown": 0, "profit_factor": None},
    }])
    efficiency = result.efficiency()
    assert efficiency["verdict"] == "INSUFFICIENT_DATA"
    assert efficiency["efficiency_ratio"] is None


def _folds(is_exp, oos_exp, trades=10):
    return [{
        "fold": i + 1,
        "is": {"trades": trades, "win_rate": 0.5, "expectancy_r": a,
               "net_pnl": 0, "max_drawdown": 0, "profit_factor": None},
        "oos": {"trades": trades, "win_rate": 0.5, "expectancy_r": b,
                "net_pnl": 0, "max_drawdown": 0, "profit_factor": None},
    } for i, (a, b) in enumerate(zip(is_exp, oos_exp, strict=True))]


def test_efficiency_flags_a_strategy_that_failed_out_of_sample():
    result = WalkForwardResult(folds=_folds([1.0, 1.2, 0.9], [-0.1, -0.2, -0.05]))
    assert result.efficiency()["verdict"] == "FAILED_OUT_OF_SAMPLE"


def test_efficiency_flags_overfitting():
    result = WalkForwardResult(folds=_folds([1.0, 1.2, 0.9], [0.1, 0.15, 0.05]))
    efficiency = result.efficiency()
    assert efficiency["verdict"] == "LIKELY_OVERFIT"
    assert efficiency["efficiency_ratio"] < 0.4


def test_efficiency_recognises_a_result_that_generalised():
    result = WalkForwardResult(folds=_folds([0.5, 0.6, 0.4], [0.45, 0.5, 0.42]))
    efficiency = result.efficiency()
    assert efficiency["verdict"] == "GENERALISED"
    assert efficiency["efficiency_ratio"] > 0.7
    assert "not evidence of a durable edge" in efficiency["note"]


def test_efficiency_flags_inconsistency_across_folds():
    result = WalkForwardResult(folds=_folds([0.5] * 4, [1.6, -0.3, -0.2, -0.1]))
    assert result.efficiency()["verdict"] in ("INCONSISTENT", "LIKELY_OVERFIT")


def test_summary_names_the_folds_it_excluded():
    folds = _folds([1.0, 1.0], [0.9, 0.9])
    folds.append({"fold": 3,
                  "is": {"trades": 1, "win_rate": 0, "expectancy_r": 0,
                         "net_pnl": 0, "max_drawdown": 0, "profit_factor": None},
                  "oos": {"trades": 0, "win_rate": 0, "expectancy_r": 0,
                          "net_pnl": 0, "max_drawdown": 0, "profit_factor": None}})
    summary = WalkForwardResult(folds=folds).summary()
    assert "excluded from the comparison" in summary


@pytest.mark.slow
def test_walk_forward_runs_end_to_end():
    from quantsutra.data import generate_index_series

    # train + test must fit inside the history, and each fold must exceed the
    # warmup -- 900 bars gives two folds at 400/300.
    frame = generate_index_series(days=900, seed=11)
    config = BacktestConfig(mode="index", warmup_bars=250, initial_capital=2_000_000)
    result = walk_forward(frame, config, train_bars=400, test_bars=300)
    assert result.folds
    for fold in result.folds:
        assert fold["is"]["trades"] >= 0
        assert "verdict" not in fold
    assert result.efficiency()["verdict"] in (
        "GENERALISED", "LIKELY_OVERFIT", "FAILED_OUT_OF_SAMPLE",
        "INCONSISTENT", "INSUFFICIENT_DATA")


# --- paper broker ---------------------------------------------------------

def test_market_order_crosses_the_spread_against_you():
    broker = PaperBroker(slippage_pct=0.004)
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    assert broker.fills[0].price == pytest.approx(200.8)
    broker.place_order(Order("X", "SELL", 75, OrderType.MARKET), market_price=200.0)
    assert broker.fills[1].price == pytest.approx(199.2)


def test_market_order_without_a_price_is_rejected():
    order = PaperBroker().place_order(Order("X", "BUY", 75, OrderType.MARKET))
    assert order.status == "REJECTED"
    assert "market_price" in order.reject_reason


def test_limit_order_fills_only_when_price_trades_through():
    broker = PaperBroker()
    broker.place_order(Order("X", "BUY", 75, OrderType.LIMIT, price=100.0))
    assert broker.on_price("X", 105, high=106, low=101) == []
    fills = broker.on_price("X", 99, high=104, low=98)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0)


def test_stop_orders_slip_past_their_trigger():
    """A triggered stop becomes a market order, so it pays the spread. This is
    why realised stop losses are worse than the level suggests."""
    broker = PaperBroker(slippage_pct=0.004)
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    broker.place_order(Order("X", "SELL", 75, OrderType.SL_M, trigger_price=180.0))
    fills = broker.on_price("X", 175.0, high=201, low=170)
    assert fills[0].price < 180.0


def test_pnl_accounting_is_consistent():
    broker = PaperBroker(starting_capital=500_000, slippage_pct=0.0)
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    broker.place_order(Order("X", "SELL", 75, OrderType.MARKET), market_price=260.0)
    funds = broker.funds()
    assert funds["realised_pnl"] == pytest.approx(60 * 75)
    assert funds["open_positions"] == 0
    # Equity must equal starting capital plus P&L minus charges.
    assert funds["equity"] == pytest.approx(
        500_000 + funds["realised_pnl"] - funds["charges_paid"], rel=1e-6)


def test_charges_are_applied_on_every_fill():
    broker = PaperBroker()
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    assert broker.fills[0].charges > 0


def test_averaging_into_a_position():
    broker = PaperBroker(slippage_pct=0.0)
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=100.0)
    position = broker.positions()[0]
    assert position.quantity == 150
    assert position.average_price == pytest.approx(150.0)


def test_flatten_closes_everything():
    broker = PaperBroker()
    broker.place_order(Order("X", "BUY", 75, OrderType.MARKET), market_price=200.0)
    broker.flatten({"X": 210.0})
    assert broker.positions() == []


def test_zero_quantity_is_rejected():
    order = PaperBroker().place_order(Order("X", "BUY", 0, OrderType.MARKET),
                                      market_price=100.0)
    assert order.status == "REJECTED"

"""Signal engine: scoring, vetoes, risk geometry and end-to-end analysis."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantsutra.analysis import AnalysisConfig, analyze
from quantsutra.constants import IST, Action, Bias, Direction, score_to_bias
from quantsutra.signals.base import CATEGORY_CAPS, Category, RuleResult
from quantsutra.signals.engine import EngineConfig, SignalEngine, build_context
from quantsutra.signals.playbooks import match_playbooks
from quantsutra.signals.rules import ALL_RULES, evaluate_rules

NOW = dt.datetime(2025, 9, 10, 11, 30, tzinfo=IST)


def test_bias_bucketing_is_symmetric():
    assert score_to_bias(0.9) == Bias.STRONG_BULL
    assert score_to_bias(-0.9) == Bias.STRONG_BEAR
    assert score_to_bias(0.0) == Bias.NEUTRAL
    assert score_to_bias(0.3) == Bias.BULL
    assert score_to_bias(-0.3) == Bias.BEAR


def test_every_rule_returns_a_result_or_a_zero_confidence_stub(uptrend):
    ctx = build_context("NIFTY", uptrend, "1d", now=NOW)
    results = evaluate_rules(ctx)
    assert len(results) == len(ALL_RULES)
    for result in results:
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.rationale, f"{result.name} returned no rationale"
        assert result.category in vars(Category).values()


def test_a_broken_rule_cannot_kill_the_scan(uptrend):
    def exploding_rule(ctx):
        raise ValueError("boom")

    ctx = build_context("NIFTY", uptrend, "1d", now=NOW)
    results = evaluate_rules(ctx, [exploding_rule] + ALL_RULES[:3])
    assert len(results) == 4
    assert results[0].confidence == 0.0
    assert "ValueError" in results[0].rationale


def test_zero_confidence_rules_are_excluded_from_scoring():
    engine = SignalEngine()
    live = [RuleResult("a", Category.TREND, 1.0, 1.0, 1.0, "bullish")]
    with_dead = live + [RuleResult("b", Category.MOMENTUM, 0.0, 1.0, 0.0, "no data")]
    assert engine.score(live)[0] == pytest.approx(engine.score(with_dead)[0])


def test_category_caps_prevent_one_family_from_dominating():
    """Eight correlated trend rules must not outweigh everything else."""
    engine = SignalEngine()
    many_trend = [RuleResult(f"t{i}", Category.TREND, 1.0, 1.0, 1.0, "up") for i in range(8)]
    one_structure = [RuleResult("s", Category.STRUCTURE, -1.0, 1.0, 1.0, "down")]
    score, categories = engine.score(many_trend + one_structure)
    assert categories[Category.TREND] == pytest.approx(1.0)
    assert categories[Category.STRUCTURE] == pytest.approx(-1.0)
    # With caps of 0.22 (trend) and 0.20 (structure), eight trend rules net only
    # slightly positive -- not eight times as confident.
    assert abs(score) < 0.15


def test_score_is_bounded():
    engine = SignalEngine()
    extreme = [RuleResult(f"r{i}", cat, 1.0, 5.0, 1.0, "x")
               for i, cat in enumerate(CATEGORY_CAPS)]
    score, _ = engine.score(extreme)
    assert -1.0 <= score <= 1.0


def test_engine_never_leans_the_wrong_way(uptrend, downtrend):
    """The score must lean with the trend, and the engine must never call the
    opposite direction.

    NEUTRAL is a legitimate verdict even in a trend -- the engine deliberately
    stands aside when the most recent swings have gone flat -- so the invariant
    tested is the sign of the score and the absence of a wrong-way call, not
    that a trade is always produced.
    """
    engine = SignalEngine()
    up = engine.generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    down = engine.generate(build_context("NIFTY", downtrend, "1d", now=NOW))
    assert up.raw_score > 0
    assert down.raw_score < 0
    assert up.direction != Direction.DOWN
    assert down.direction != Direction.UP


def test_a_strong_clean_trend_produces_a_directional_call():
    """Given unambiguous trend data the engine must actually commit."""
    for sign, expected in ((1, Direction.UP), (-1, Direction.DOWN)):
        n = 400
        rng = np.random.default_rng(5)
        close = 20000 * np.exp(sign * np.linspace(0, 0.45, n)
                               + np.cumsum(rng.normal(0, 0.002, n)))
        data = pd.DataFrame({
            "open": close, "high": close * 1.003, "low": close * 0.997,
            "close": close, "volume": rng.integers(3e5, 9e5, n),
        }, index=pd.date_range("2023-01-02 15:30", periods=n, freq="B", tz=IST))
        signal = SignalEngine().generate(build_context("NIFTY", data, "1d", now=NOW))
        assert signal.direction == expected, (
            f"expected {expected} on a clean trend, got {signal.direction} "
            f"(score {signal.raw_score:+.3f})"
        )


def test_engine_stands_aside_on_noise():
    """A pure random walk should not produce a confident directional call."""
    rng = np.random.default_rng(99)
    n = 400
    close = 22000 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    data = pd.DataFrame({
        "open": close, "high": close * 1.004, "low": close * 0.996,
        "close": close, "volume": rng.integers(2e5, 9e5, n),
    }, index=pd.date_range("2023-01-02 15:30", periods=n, freq="B", tz=IST))
    signal = SignalEngine().generate(build_context("NIFTY", data, "1d", now=NOW))
    assert abs(signal.raw_score) < 0.6


def test_targets_are_r_multiples_of_actual_risk(uptrend):
    signal = SignalEngine().generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    if signal.entry is None:
        pytest.skip("no directional signal on this fixture")
    risk = abs(signal.entry - signal.stop_loss)
    assert risk > 0
    for target in signal.targets:
        assert abs(target - signal.entry) / risk >= 0.5


def test_stop_is_capped_as_a_share_of_spot(uptrend):
    config = EngineConfig(max_stop_pct=0.02)
    signal = SignalEngine(config).generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    if signal.stop_loss is None:
        pytest.skip("no directional signal on this fixture")
    assert abs(signal.entry - signal.stop_loss) <= signal.spot * 0.02 + 1e-6


def test_targets_are_clamped_to_a_plausible_distance(uptrend):
    config = EngineConfig(max_target_pct=0.05)
    signal = SignalEngine(config).generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    if not signal.targets:
        pytest.skip("no directional signal on this fixture")
    for target in signal.targets:
        assert abs(target - signal.spot) <= signal.spot * 0.05 + 1e-6


def test_counter_higher_timeframe_trades_are_vetoed(uptrend, downtrend):
    """A long against a firmly bearish weekly must be blocked."""
    from quantsutra.data.base import resample_ohlcv
    bearish_weekly = resample_ohlcv(downtrend, "1W")
    ctx = build_context("NIFTY", uptrend, "1d", higher_tf_ohlcv=bearish_weekly, now=NOW)
    signal = SignalEngine().generate(ctx)
    if signal.direction == Direction.UP:
        assert any("Higher timeframe" in v for v in signal.vetoes)


def test_signal_dict_is_json_safe(uptrend):
    import json
    signal = SignalEngine().generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    json.dumps(signal.to_dict(), default=str)
    assert signal.to_dict()["confidence"] >= 0


def test_reasons_are_populated(uptrend):
    signal = SignalEngine().generate(build_context("NIFTY", uptrend, "1d", now=NOW))
    assert signal.reasons
    assert all(isinstance(reason, str) and reason for reason in signal.reasons)


def test_playbooks_declare_their_failure_mode(uptrend):
    ctx = build_context("NIFTY", uptrend, "1d", now=NOW)
    for match in match_playbooks(ctx):
        assert match.fails_when, f"{match.name} does not say when it fails"
        assert match.horizon


def test_playbook_stops_are_a_sane_distance_from_price(synthetic):
    """A playbook must never pair today's price with a far-away stale level."""
    for i in range(300, len(synthetic), 40):
        ctx = build_context("NIFTY", synthetic.iloc[i - 260:i], "1d", now=NOW)
        close = float(ctx.get("close"))
        for match in match_playbooks(ctx):
            if match.invalidation is None:
                continue
            assert abs(close - match.invalidation) / ctx.atr < 8


def test_analyze_returns_a_complete_result(synthetic, option_chain):
    result = analyze("NIFTY", synthetic, "1d", chain=option_chain, now=NOW,
                     config=AnalysisConfig(capital=1_500_000))
    for key in ("signal", "recommendation", "levels", "event_risk", "disclaimer",
                "structure", "playbooks", "historical_lessons"):
        assert key in result, f"{key} missing from analyze output"
    assert result["recommendation"]["action"] in [a.value for a in Action]


def test_analyze_refuses_insufficient_data(uptrend, option_chain):
    result = analyze("NIFTY", uptrend.head(30), "1d", chain=option_chain, now=NOW)
    assert "error" in result
    assert "60" in result["error"]


def test_analyze_works_without_an_option_chain(synthetic):
    result = analyze("NIFTY", synthetic, "1d", now=NOW)
    assert "signal" in result
    assert "contract" not in result or result.get("contract_note")


def test_analyze_output_is_json_serialisable(synthetic, option_chain):
    import json
    result = analyze("NIFTY", synthetic, "1d", chain=option_chain, now=NOW)
    json.dumps(result, default=str)


def test_disclaimer_is_present_and_honest(synthetic):
    result = analyze("NIFTY", synthetic, "1d", now=NOW)
    text = result["disclaimer"].lower()
    assert "not investment advice" in text
    assert "not a prediction" in text


def _trending_frame(seed=5, n=400, slope=0.45):
    """A clean uptrend the engine reliably calls, so the assertions below are
    about the recommendation text rather than about whether a signal fired."""
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.linspace(0, slope, n) + np.cumsum(rng.normal(0, 0.002, n)))
    frame = pd.DataFrame({
        "open": close, "high": close * 1.003, "low": close * 0.997,
        "close": close, "volume": rng.integers(3e5, 9e5, n),
    }, index=pd.date_range("2023-01-02 15:30", periods=n, freq="B", tz=IST))
    return frame


def _chain_for(spot, dte=6, step=50, lot_size=75):
    import datetime as _dt

    from quantsutra.options.chain import chain_from_records
    from quantsutra.options.pricing import bs_price, time_to_expiry

    t = time_to_expiry(dte)
    base = round(spot / step) * step
    records = []
    for strike in np.arange(base - 2000, base + 2050, step):
        strike = float(strike)
        m = (strike - spot) / spot
        iv = 0.125 + 0.9 * m**2 - 0.55 * m
        oi = int(450_000 * np.exp(-((abs(m) * 100) ** 2) / 9)) + 30_000
        records += [
            {"strike": strike, "type": "CE", "ltp": round(bs_price(spot, strike, t, iv, "CE"), 2),
             "oi": oi, "iv": round(iv * 100, 2), "volume": oi // 3, "oi_change": 8000},
            {"strike": strike, "type": "PE", "ltp": round(bs_price(spot, strike, t, iv, "PE"), 2),
             "oi": int(oi * 1.18), "iv": round(iv * 100, 2), "volume": oi // 2, "oi_change": 12000},
        ]
    return chain_from_records("NIFTY", spot, _dt.date(2025, 9, 16), records,
                              dte_trading=dte, lot_size=lot_size, strike_step=step)


def test_no_chain_does_not_blame_the_risk_limit():
    """With no option chain nothing is sized, so the recommendation must say
    that -- not report 'position size is ZERO at your risk limit', which blames
    the wrong thing entirely."""
    frame = _trending_frame()
    result = analyze("NIFTY", frame, "1d", now=NOW,
                     config=AnalysisConfig(capital=10_000_000))
    assert result["signal"]["direction"] == "UP", "fixture should produce a signal"
    summary = result["recommendation"]["summary"]
    assert "No option chain was supplied" in summary
    assert "risk limit" not in summary
    assert result.get("position") is None


def test_zero_size_reports_the_capital_it_would_need():
    """A zero-lot answer is only useful if it says what would make it non-zero."""
    frame = _trending_frame()
    chain = _chain_for(float(frame["close"].iloc[-1]))
    result = analyze("NIFTY", frame, "1d", chain=chain, now=NOW,
                     config=AnalysisConfig(capital=100_000, risk_pct=0.01))
    position = result["position"]
    assert position["lots"] == 0, "one lot cannot fit a 1% risk budget on Rs.1 lakh"
    needed = position["capital_needed_for_one_lot"]
    assert needed > 100_000
    assert "ZERO at your risk limit" in result["recommendation"]["summary"]
    assert f"Rs.{needed:,.0f}" in result["recommendation"]["details"][0]


def test_size_scales_with_capital():
    """The same signal must size up as capital grows, staying inside the limit."""
    frame = _trending_frame()
    chain = _chain_for(float(frame["close"].iloc[-1]))
    previous = 0
    for capital in (1_500_000, 4_000_000, 10_000_000):
        result = analyze("NIFTY", frame, "1d", chain=chain, now=NOW,
                         config=AnalysisConfig(capital=capital, risk_pct=0.01))
        position = result["position"]
        assert position["lots"] >= previous, "more capital must not mean fewer lots"
        assert position["risk_pct_of_capital"] <= 0.0101, "risk must stay inside the limit"
        previous = position["lots"]
    assert previous >= 2, "the largest account should take more than one lot"

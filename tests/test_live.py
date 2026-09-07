"""Journal, notifier and the live scanner loop (no network)."""

import datetime as dt

import pytest

from quantsutra.live import Journal, Notifier, TelegramSink, format_signal
from quantsutra.live.notify import ConsoleSink
from quantsutra.live.runner import LiveScanner, ScannerConfig


class SilentSink:
    name = "silent"

    def __init__(self):
        self.messages = []

    def send(self, text, title=""):
        self.messages.append((title, text))
        return True


@pytest.fixture
def journal(tmp_path):
    with Journal(tmp_path / "j.sqlite") as j:
        yield j


def _signal(**overrides):
    base = {
        "symbol": "NIFTY", "timeframe": "15m", "direction": "UP", "action": "BUY_CALL",
        "confidence": 70.0, "raw_score": 0.35, "spot": 25000.0, "entry": 25000.0,
        "stop_loss": 24850.0, "targets": [25225.0], "regime": {"regime": "UPTREND"},
        "actionable": True, "vetoes": [], "reasons": ["trend up"],
        "timestamp": "2025-09-08T11:00:00",
    }
    base.update(overrides)
    return base


def test_journal_logs_no_trade_signals_too(journal):
    """A system that only records its opinions when it has one cannot be
    evaluated honestly."""
    journal.log_signal(_signal(action="NO_TRADE", actionable=False))
    journal.log_signal(_signal())
    assert len(journal.recent()) == 2
    assert journal.hit_rate()["actionable_signals"] == 1


def test_journal_computes_r_multiples(journal):
    signal_id = journal.log_signal(_signal())
    journal.record_outcome(signal_id, 25225.0, "TARGET")
    stats = journal.hit_rate()
    assert stats["resolved"] == 1
    assert stats["avg_r"] == pytest.approx(1.5, abs=0.01)


def test_journal_counts_unresolved_separately(journal):
    for _ in range(5):
        journal.log_signal(_signal())
    resolved_id = journal.log_signal(_signal())
    journal.record_outcome(resolved_id, 25225.0, "TARGET")
    stats = journal.hit_rate()
    assert stats["unresolved"] == 5
    assert stats["resolved"] == 1


def test_journal_reports_a_confidence_interval_and_warns_on_small_samples(journal):
    for i in range(8):
        signal_id = journal.log_signal(_signal())
        journal.record_outcome(signal_id, 25225.0 if i % 2 else 24850.0,
                               "TARGET" if i % 2 else "STOP")
    stats = journal.hit_rate()
    assert stats["hit_rate_ci"][0] < stats["hit_rate"] < stats["hit_rate_ci"][1]
    assert "too wide to mean anything" in stats["note"]


def test_journal_breaks_results_down_by_regime(journal):
    for i in range(6):
        regime = "UPTREND" if i % 2 else "RANGE"
        signal_id = journal.log_signal(_signal(regime={"regime": regime}))
        journal.record_outcome(signal_id, 25225.0, "TARGET")
    by_regime = {row["regime"]: row for row in journal.stats_by_regime()}
    assert set(by_regime) == {"UPTREND", "RANGE"}
    assert sum(row["signals"] for row in by_regime.values()) == 6


def test_notifier_suppresses_an_unchanged_view():
    notifier = Notifier(sinks=[], cooldown_minutes=45)
    now = dt.datetime(2025, 9, 8, 11, 0)
    assert notifier.should_send("NIFTY", "BUY_CALL", now)
    notifier._last["NIFTY"] = ("BUY_CALL", now)
    assert not notifier.should_send("NIFTY", "BUY_CALL", now + dt.timedelta(minutes=10))
    assert notifier.should_send("NIFTY", "BUY_PUT", now + dt.timedelta(minutes=10))
    assert notifier.should_send("NIFTY", "BUY_CALL", now + dt.timedelta(minutes=50))


def test_notifier_skips_no_trade_by_default():
    notifier = Notifier(sinks=[])
    assert not notifier.should_send("NIFTY", "NO_TRADE")
    assert Notifier(sinks=[], notify_no_trade=True).should_send("NIFTY", "NO_TRADE")


def test_notifier_delivers_to_its_sinks():
    sink = SilentSink()
    notifier = Notifier(sinks=[sink])
    result = notifier.send({"signal": _signal(), "recommendation": {"summary": "go"}})
    assert result["sent"]
    assert sink.messages and "BUY_CALL" in sink.messages[0][0]


def test_telegram_is_unconfigured_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    sink = TelegramSink()
    assert not sink.configured
    assert sink.send("hello") is False


def test_formatted_signal_carries_the_essentials_and_a_caveat():
    text = format_signal({
        "signal": _signal(risk_reward=1.5),
        "contract": {"strikes": {"long_ce": 25100.0}},
        "position": {"lots": 2, "risk_amount": 9800},
        "recommendation": {"summary": "BUY_CALL: 2 lots"},
    }, verbose=True)
    assert "NIFTY" in text and "BUY_CALL" in text
    assert "25100" in text
    assert "not advice" in text.lower()


class FakeFeed:
    name = "fake"

    def __init__(self, frame):
        self.frame = frame

    def history(self, symbol, interval="1d", lookback=500, start=None, end=None):
        return self.frame.tail(lookback)

    def quote(self, symbol):
        return {"symbol": symbol, "price": float(self.frame["close"].iloc[-1])}


def test_scanner_runs_one_pass_and_journals_it(tmp_path, synthetic):
    sink = SilentSink()
    config = ScannerConfig(symbols=("NIFTY",), timeframe="1d", fetch_chain=False,
                           journal_path=str(tmp_path / "j.sqlite"), max_iterations=1)
    scanner = LiveScanner(config, feed=FakeFeed(synthetic), notifier=Notifier(sinks=[sink]))
    try:
        results = scanner.scan_once(dt.datetime(2025, 9, 8, 11, 30))
        assert len(results) == 1
        assert "error" not in results[0]
        assert results[0]["journal_id"] > 0
        assert scanner.journal.recent()
    finally:
        scanner.close()


def test_scanner_survives_a_broken_feed(tmp_path):
    class BrokenFeed:
        name = "broken"

        def history(self, *args, **kwargs):
            raise RuntimeError("feed down")

    config = ScannerConfig(symbols=("NIFTY", "BANKNIFTY"), fetch_chain=False,
                           journal_path=str(tmp_path / "j.sqlite"))
    scanner = LiveScanner(config, feed=BrokenFeed(), notifier=Notifier(sinks=[]))
    try:
        results = scanner.scan_once(dt.datetime(2025, 9, 8, 11, 30))
        assert len(results) == 2
        assert all("error" in r for r in results), "one bad symbol must not stop the loop"
    finally:
        scanner.close()

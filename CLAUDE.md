# quantsutra — working notes

Analysis and signal engine for Indian index derivatives. Read `README.md` for
what it is, `docs/ARCHITECTURE.md` for how it fits together, and
`docs/LIMITATIONS.md` before trusting any output.

## Commands

```bash
pip install -e ".[dev]"
pytest -q -m "not slow"     # ~6s
pytest -q                   # ~4.5 min (full backtests)
ruff check quantsutra tests
quantsutra demo             # whole pipeline, offline
```

## Conventions that matter here

**Exchange rules are dated config, never hardcoded.** Lot sizes, expiry
weekdays and tax rates live in `config/*.yaml` keyed by effective date. If you
find yourself writing `if symbol == "NIFTY": lot = 75`, put it in
`config/instruments.yaml` instead — a backtest over 2023 must resolve 2023's
rules.

**Indicators must be causal.** Nothing may read a bar later than the one being
evaluated. `tests/test_indicators.py::test_compute_all_produces_a_wide_frame_without_lookahead`
enforces this by truncating the input and asserting earlier values are
unchanged. Forward-shifted series (Ichimoku's chikou, unconfirmed fractals) are
fine only because they are NaN at the evaluation bar.

**A rule with no data returns `confidence=0`, never a neutral score.** The
engine drops zero-confidence rules; a neutral vote would dilute real evidence.

**New rules need a family, and the family cap is the point.** `CATEGORY_CAPS`
in `signals/base.py` is what stops eight correlated trend indicators from
manufacturing 95% confidence out of one fact. Adding a rule to an
already-crowded family is usually right; inventing a new family to give an idea
more weight is usually wrong.

**Playbooks must declare `fails_when`.** A setup whose failure mode you cannot
describe is one you will keep taking after it stops working.

**Report the unflattering number.** Win-rate confidence intervals, cost drag,
dependence on a single trade, skipped-signal tallies — these stay in the
output. If a change makes results look better, check it did not do so by
hiding something.

**No live order placement.** `brokers/` ships a paper broker only. That is a
deliberate design decision, not a missing feature.

## Testing style

Assert mathematical identities and invariants, not snapshots — put-call parity,
the delta relation, IV round-trips, "the score leans with the trend", "wider
bands contain more bars". Snapshot tests here rot immediately and catch
nothing. When a test fails, check whether the test's expectation or the code is
wrong; several tests in this suite were fixed because the code was right and
the fixture was not.

Mark anything that runs a full backtest `@pytest.mark.slow`.

## Gotchas

- `ensure_ohlcv` returns the input uncopied when it is already canonical.
  Callers must treat the result as read-only.
- `df.attrs` does not survive `.join()`; `compute_all` stamps it last.
- pandas comparisons between differently-indexed Series raise in pandas 3 —
  align with a boolean mask, not `.loc[other.index]`.
- The NSE and Yahoo endpoints are blocked from most cloud hosts. Use
  `--source synthetic` or `CsvFeed` when developing without a home connection.

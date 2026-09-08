# Architecture

## Design principles

**Every layer degrades rather than guesses.** No option chain means no strike
recommendation, but direction and levels still work. No volume means volume
rules are skipped, not filled with zeros. A rule that cannot evaluate returns
confidence 0 and is dropped, rather than voting "neutral" and diluting real
evidence.

**Exchange rules are dated data, not code.** Lot sizes and expiry weekdays have
changed repeatedly. They live in `config/instruments.yaml` keyed by the date
each rule took effect, so a 2023 backtest resolves 2023's Thursday expiry and a
2025 one resolves Tuesday.

**Correlated evidence is capped, not summed.** Eight moving-average rules
agreeing is one fact counted eight times. Family caps are the mechanism that
stops this from manufacturing confidence.

**The unflattering number is reported.** Win-rate confidence intervals, cost
drag, dependence on a single trade, time in market, and the reasons signals
were skipped are all in the output by default.

---

## Layer map

```
config/*.yaml            dated exchange rules you own and update
    │
    ▼
calendar_in              sessions, holidays, expiry resolution
    │
data/                    yahoo | nse | csv | synthetic, + validation + cache
    │
    ▼
indicators/              ~60 indicators, all causal (no forward shift used at t)
    │  trend  momentum  volatility  volume  pivots
    │
patterns/                swings → candles, classical, levels, structure
    │
    ▼
regime                   orthogonal inputs → one of 7 regime labels
    │
options/                 pricing → chain → strategies → selection
    │
    ▼
signals/                 base (vocabulary) → rules (32) → engine (confluence
    │                    + vetoes) → playbooks (named setups)
    │
risk/                    costs → sizing → manager (portfolio limits)
    │
    ▼
analysis.analyze()       orchestration; the only function most callers need
    │
    ├──► cli/            eight commands
    ├──► report/         rich terminal rendering
    └──► backtest/       walk-forward simulation over the same pipeline
```

`knowledge/` sits alongside and is read by `analysis` for event risk and
historical context.

---

## The scoring pipeline in detail

### 1. Context assembly (`signals.engine.build_context`)

Computed once per evaluation so no rule recomputes an indicator and no rule can
see a bar that has not closed:

- `compute_all(ohlcv)` → ~130 feature columns
- `snapshot(features)` → flat, JSON-safe dict of the latest bar
- regime classification, market structure, candlestick scan, chart patterns
- level map (swing clusters + volatility-scaled round numbers), plus a
  separate *strong-levels-only* map used for vetoes
- fair value gaps, order blocks, liquidity sweeps
- higher-timeframe snapshot, option chain summary, expiry context, session phase

### 2. Rule evaluation (`signals.rules`)

32 rules across nine families. Each returns:

| Field | Meaning |
|---|---|
| `score` | −1 to +1, the rule's directional opinion |
| `weight` | importance within its family |
| `confidence` | 0 when the rule lacked data (dropped entirely) |
| `rationale` | plain-English explanation, shown to the user |

A rule that raises is caught and recorded as a zero-confidence stub — one
broken rule cannot kill a scan.

### 3. Confluence (`SignalEngine.score`)

1. Average each family's rules, weighted by `weight × confidence`.
2. Combine families, each capped by `CATEGORY_CAPS`.
3. Normalise by the caps that actually had data, so a missing option chain
   does not silently deflate every score.

Confidence blends magnitude (40%), cross-family agreement (30%), breadth (15%)
and data quality (15%). **It is not a probability of profit** and has not been
calibrated against outcomes.

### 4. Vetoes (`SignalEngine.vetoes`)

Hard blocks no confluence can override:

- trading against a firmly opposed higher timeframe
- less than 0.4 ATR of room to a *strong* level in the trade's direction
- volatile-chop regime
- a squeeze with no established direction
- reward/risk to T1 below the floor
- insufficient history

### 5. Risk geometry (`SignalEngine.levels_for`)

Stop first, everything derived from it:

1. Wider of an ATR stop and the structural invalidation (swing) — a stop inside
   the noise band gets taken out even when the idea is right.
2. Capped by the *tighter* of `max_stop_atr_mult × ATR` and `max_stop_pct × spot`.
3. Targets at R-multiples of the resulting risk, so reward/risk is consistent
   by construction rather than an accident of stop width.
4. T1 snapped to a strong level if one sits in the way — which lowers the
   reported reward/risk, and correctly vetoes an entry into a wall.
5. Targets clamped to `max_target_pct` of spot, flagged when it binds.

### 6. Structure, strike, size

- `choose_structure(direction, iv_percentile, dte, trend_strength)` decides
  *how* to express the view. Rich IV plus a strong trend → debit spread. Cheap
  IV → naked long. No direction plus rich IV and time → iron condor.
- `select_strike` targets a delta, filters on open interest, and says so.
- `size_option_position` takes the more conservative of a delta-translated
  index stop and a premium stop, then caps total premium outlay — because an
  overnight gap takes the whole premium, not just the stop distance.

---

## Look-ahead safety

The backtester rebuilds the entire context from `ohlcv[:i+1]` each bar and
fills at bar `i+1`'s open. This is slower than vectorising, but a vectorised
feature frame leaks the future through any centred or forward-shifted
calculation, and that leak is exactly what makes a backtest irreproducible
live.

Two forward-looking calculations exist inside the indicator set and are safe
because they are always NaN at the evaluation bar:

- `ichimoku.chikou` is `close.shift(-26)`
- fractals/swings require `right` bars of confirmation

`tests/test_indicators.py::test_compute_all_produces_a_wide_frame_without_lookahead`
asserts this directly: truncating the input must not change any earlier value.

When a bar touches both the stop and the target, the engine assumes the **stop**
filled first. Without intrabar data the order is unknowable, and assuming the
favourable one is how backtests quietly inflate win rates.

---

## Extending it

**A new indicator** — add it to the right family module in `indicators/`,
export it from `indicators/__init__.py`, and wire it into `compute_all` if it
should be in the standard feature frame.

**A new rule** — write a function taking `MarketContext` and returning
`RuleResult`, then append it to `ALL_RULES`. Return `confidence=0` when the
inputs are missing. Pick the family carefully: the family cap is what stops it
from double-counting existing evidence.

**A new playbook** — write a detector returning `PlaybookMatch | None` and add
a `Playbook` entry declaring its valid regimes, session phases, and — required
— its `fails_when` string. A setup whose failure mode you cannot describe is
one you will keep taking after it stops working.

**A new data feed** — implement `history()` and `quote()` per the `DataFeed`
protocol in `data/base.py`, return frames through `normalise_frame`, and
register it in `data.get_feed`.

**A broker integration** — there is deliberately none. `brokers/` is left as a
place to add one behind a paper-trading adapter first.

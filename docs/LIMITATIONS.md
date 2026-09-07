# Limitations

An honest accounting of where this tool is weak, wrong, or misleading. Read it
before you act on any output.

## 1. It cannot predict prices

No configuration of this software will reliably forecast index direction. It
applies published technical and options methods — patterns that everyone can
see, which is precisely why any edge in them is small and unstable. There is no
proprietary data, no order flow, no news, no fundamental input.

The `confidence` number measures **how much the evidence agrees with itself**,
not the probability of profit. A 90-confidence signal is one where nine
families of evidence point the same way. Markets regularly go the other way
anyway.

## 2. Synthetic option pricing in backtests

`--mode options` prices options with Black-Scholes from the index path and a
flat assumed IV, because historical Indian option chains are not freely
available. What that omits:

| Missing | Effect on results |
|---|---|
| Volatility smile | OTM options priced too cheaply → OTM buying looks better than it is |
| IV path | No post-event IV crush, the most common way a correct call still loses |
| Real bid-ask | Modelled as a flat percentage; actual spreads are wider and vary |
| Early exercise | Not applicable — Indian index options are European, so this one is fine |

**Options-mode backtest results are an upper bound, not an expectation.** Use
`--mode index` when you want to know whether the *signal* has an edge, then
discount heavily for the option overlay.

## 3. Small samples cannot establish an edge

Any backtest here produces dozens to a few hundred trades. A 60% win rate over
30 trades has a 95% confidence interval of roughly 42%–75% — which includes
"no edge at all". The metrics report this interval and flag when it straddles
50%, but no amount of reporting turns a small sample into evidence.

## 4. Contract specifications drift

Lot sizes and expiry weekdays are set by exchange circular and changed several
times in 2024–2025 alone. `config/instruments.yaml` carries a `verified_on`
date and effective-dated rules, but **the values were compiled from public
record, not fetched live**. A stale lot size makes every position size wrong by
a whole multiple. Verify against the exchange before trading.

The same applies to `config/costs.yaml` (STT rates changed in October 2024) and
`config/holidays.yaml` (published annually; unverified years are flagged
`provisional` and the calendar warns when it uses one).

## 5. Data feed limitations

- **Yahoo** — unofficial endpoint, delayed, ~60 days of intraday history, and
  index volume is frequently zero or missing. No uptime guarantee.
- **NSE** — undocumented endpoints, aggressive rate limiting, blocks datacentre
  IPs and many VPNs. Will generally not work from a cloud host. The chain is a
  delayed snapshot, not an execution quote.
- **Both** can change their response shape without notice.

For real money, export history from your broker and use `CsvFeed`.

## 6. Index volume is unreliable

NIFTY and SENSEX spot volume from free feeds is often zero. The package detects
this and **skips volume-based rules entirely** rather than computing OBV on
zeros. That is the right behaviour, but it means the volume family contributes
nothing on those feeds. Use futures volume or an index ETF if volume matters to
you.

## 7. No execution, deliberately

There is no order placement and no broker integration. An uncalibrated signal
engine wired to live orders destroys accounts faster than manual trading does,
not slower. Paper trade first, for months.

## 8. Max pain is weak evidence

It is computed and reported because it is widely watched, but the academic
result is that it has little predictive power outside the last hours of expiry.
The signal engine gives it a small weight and only when DTE ≤ 1, and the
rationale says so.

## 9. Historical analogues are not forecasts

`find_analogues` reports past windows whose *shape* resembles the current one.
Financial series are noisy enough that striking matches occur by chance; the
forward returns come from a sample usually in single digits; and markets are
non-stationary — the conditions of 2008 do not exist now, however similar the
chart looks. The function reports the dispersion and range of forward returns
alongside the mean precisely so the uncertainty stays visible.

## 10. Regime classification is a label, not truth

Seven discrete labels over a continuous, noisy process. Boundaries are
arbitrary and regimes are only identifiable in hindsight. The classifier is
useful because it stops a trend strategy from firing in a range — not because
it knows what the market is.

## 11. Costs are modelled, not measured

The statutory components (STT, GST, stamp duty, exchange charges) are accurate
if the config is current. **Slippage is a guess.** The defaults are
conservative but your actual bid-ask is the number that decides whether a
strategy works, and it is the single largest cost component for retail option
trades. If you have quote data, use it.

## 12. What SEBI's data says

Studies of individual traders in the Indian equity derivatives segment have
repeatedly found that a large majority lose money, and that losses scale with
trading frequency. This tool does not change that base rate. Its most valuable
output is often `NO_TRADE`.

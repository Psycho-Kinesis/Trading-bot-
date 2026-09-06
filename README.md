# quantsutra

A rule-based analysis and signal engine for Indian index derivatives — NIFTY,
BANKNIFTY, SENSEX and their futures and options.

It reads price and option-chain data, computes a wide indicator and pattern
surface, classifies the market regime, and produces a specific trade
recommendation: direction, option structure, strike, expiry, entry, stop,
targets and position size — with the reasoning for every part of it, and the
reasons it decided *not* to trade when it decides that.

---

## Read this first

**This will not make you reliably profitable, and no version of it will.**

You asked for signals with great accuracy and profits. That is worth being
straight about, because the gap between that and what any software can do is
where most trading capital goes:

- **Nobody can predict index direction reliably.** Not this, not a paid
  service, not an institutional desk. Public technical patterns are visible to
  everyone, so any edge in them is small, unstable, and competed away. Anyone
  advertising 90% accuracy on NIFTY options is selling something.
- **Indian retail F&O loses money in aggregate.** SEBI's own studies of
  individual traders in the equity derivatives segment have repeatedly found
  that the large majority lose, and that average losses grow with trading
  frequency. Costs and the bid-ask spread explain much of it. Look up the
  current study before you fund an account.
- **This tool has no information edge.** It sees price, volume and open
  interest. It does not see news, order flow, institutional positioning, or
  anything about the future.

What it *can* do, and does well: apply a large body of published technical and
options method consistently, without the emotional errors that cost far more
than any indicator setting; show its full reasoning so you can disagree with
it; enforce risk limits mechanically; model Indian transaction costs honestly;
and tell you when there is nothing worth trading — which is most of the time.

Treat it as an analyst that argues its case and shows its work, not an oracle.
Backtest it on your own data, paper trade it for months, and risk only capital
you can afford to lose entirely.

---

## Install

```bash
git clone <this repo> && cd Trading-bot-
pip install -e .
```

Python 3.10+. Core dependencies are numpy, pandas, scipy, requests, PyYAML,
click and rich — no TA-Lib, no compiled extensions.

## Try it without any data or network

```bash
quantsutra demo
```

Runs the entire pipeline on generated data and prints the full report:
signal, regime, evidence by family, every rule that fired, vetoes, levels,
option chain analytics, event risk and historical context.

## Real usage

```bash
# Full analysis with a live option chain and your own risk settings
quantsutra signal NIFTY --chain --capital 500000 --risk 1.0

# What the market is doing, nothing else
quantsutra regime BANKNIFTY

# Option chain: PCR, max pain, OI walls, IV skew, greeks around ATM
quantsutra chain NIFTY

# Expiries, holidays and event risk
quantsutra calendar NIFTY

# What a trade actually costs after STT, GST, stamp duty and the spread
quantsutra costs --premium 45 --lots 1 --moneyness OTM

# Walk-forward backtest with all of the above modelled
quantsutra backtest NIFTY --source csv --dir data --mode options --save trades.csv

# Historical episodes and what each one teaches
quantsutra events
```

As a library:

```python
from quantsutra import analyze
from quantsutra.data import YahooFeed, NseFeed

bars = YahooFeed().history("NIFTY", "1d", 500)
chain = NseFeed().option_chain("NIFTY")          # optional
result = analyze("NIFTY", bars, chain=chain)

print(result["recommendation"]["summary"])
for line in result["signal"]["reasons"]:
    print(" ", line)
```

---

## How it decides

```
 price bars ──► indicators ──┐
                             ├──► MarketContext ──► 32 rules ──► confluence
 option chain ──► analytics ─┘         │              (9 families, capped)
                                       │                      │
 calendar ──► expiry, session ─────────┤                      ▼
                                       │              score ∈ [-1, +1]
 patterns ──► candles, structure ──────┤              confidence 0-100
              levels, formations       │                      │
                                       ▼                      ▼
                                 regime classifier ──►    vetoes
                                                              │
                                                              ▼
                        structure choice (IV × time × trend strength)
                                          │
                        strike selection (delta-targeted, liquidity-filtered)
                                          │
                        sizing (delta-translated stop, premium cap)
                                          │
                                     recommendation
```

**Rules are grouped into nine families and each family's contribution is
capped.** This is the most important design decision in the package. Eight
moving-average rules all saying "up" is *one* piece of evidence counted eight
times; without caps that manufactures 95% confidence out of a single fact.
Trend and structure get the largest budgets, options positioning a modest one,
volatility the smallest.

**A rule that cannot evaluate returns confidence 0 and is dropped**, rather
than voting "neutral" and diluting real signals.

**Vetoes override any amount of confluence.** Trading against a firmly
opposed higher timeframe, entering half an ATR below established resistance,
a regime of high volatility with no direction, reward/risk below the floor —
each blocks the trade outright and says why.

**Targets are R-multiples of actual risk**, not fixed ATR distances. Fixed
targets against a structural stop make reward/risk an accident of how wide the
stop happened to be.

---

## What is in here

| Module | What it does |
|---|---|
| `calendar_in` | NSE/BSE sessions, holidays, and expiry resolution driven by **effective-dated rules** so a 2023 backtest uses 2023's expiry weekday and lot size |
| `indicators` | ~60 indicators across trend, momentum, volatility, volume and pivots — including Supertrend, Ichimoku, CPR, anchored VWAP, volume profile, Yang-Zhang volatility, TTM squeeze |
| `patterns` | 30+ candlestick patterns with context gating, ZigZag swings, classical formations with measured-move targets, S/R clustering, trendlines, market structure (BOS/CHoCH), fair value gaps, order blocks, liquidity sweeps |
| `regime` | Regime classification from orthogonal inputs — trend strength, directional alignment, range character, volatility level |
| `options` | Black-Scholes-Merton with dividend yield and trading-day time, full greeks including vanna/charm/vomma, Brent IV solver, PCR, max pain, OI build-up, IV skew and rank, strategy payoffs, delta-targeted strike selection |
| `signals` | 32 rules, the capped confluence engine, veto logic, and six regime-gated playbooks that each declare their own failure mode |
| `risk` | Indian transaction costs (STT, exchange charges, SEBI fees, stamp duty, GST, spread), option-aware position sizing, portfolio limits with hard daily-loss and consecutive-loss breakers |
| `backtest` | Event-driven simulation with no look-ahead, realistic fills, and metrics that report Wilson confidence intervals rather than presenting a small sample as evidence |
| `knowledge` | 15 curated Indian market episodes with the lesson each teaches, recurring calendar events keyed by their IV signature, and a historical analogue finder |
| `data` | Yahoo chart endpoint, NSE option chain, local CSV/parquet, plus a regime-switching synthetic generator and a data-quality validator |

---

## Things that will surprise you

**Position sizing usually says zero.** With NIFTY's lot size at 75, one lot of
a ₹200 option is ₹15,000 of premium. At a 1% risk limit that needs roughly
₹6 lakh of capital for a single lot. The tool says so plainly rather than
quietly recommending a trade that risks 12% of a small account. This is real
arithmetic, not conservatism.

**Costs are larger than they look.** A round trip on one NIFTY lot costs
roughly ₹100–190 all-in. On a ₹20 far-OTM weekly that is a **6%+ move in the
premium just to break even** — before expiry-day spread widening, which makes
it closer to 10%. Three scalps a day, twenty days a month, is about ₹8,500 in
costs before a single rupee of P&L:

```bash
quantsutra costs --premium 20 --lots 1 --moneyness FAR_OTM --expiry-day
```

**The engine stands aside most of the time.** On a typical run it declines
roughly 80–90% of bars. That is the design working. A system that always has
an opinion is a system with no filter.

**Backtests on random data show no edge, and the tool says so.** Metrics
report the Wilson confidence interval on the win rate and flag when it
straddles 50%, when one trade produced most of the profit, and when a strategy
is profitable gross but not net.

---

## Known limitations

These are real and you should weigh them before trusting output:

1. **Synthetic option pricing in backtests.** Historical Indian option chains
   are not freely available, so `--mode options` prices options with
   Black-Scholes from the index path and an assumed IV. There is no smile
   (OTM options are priced too cheaply), no IV path (it misses the post-event
   IV crush that turns correct calls into losses), and no real bid-ask.
   **Options-mode results are optimistic for option buying.** Use
   `--mode index` for a clean read on whether the signal itself has an edge.

2. **Confidence is not a probability.** The 0–100 number measures how much the
   evidence agrees, not the chance of profit. It has not been calibrated
   against realised outcomes. Calibrate it yourself before sizing on it.

3. **Contract specifications drift.** Lot sizes and expiry weekdays change by
   exchange circular — they changed several times in 2024–2025 alone. They
   live in `config/instruments.yaml` with the date each rule took effect and a
   `verified_on` stamp. **Verify them against the exchange before trading.**
   A stale lot size makes every position size wrong by a whole multiple.

4. **Holiday lists go stale.** `config/holidays.yaml` marks unverified years
   `provisional` and the calendar warns when it uses one. An out-of-date list
   silently shifts every expiry that lands near a holiday.

5. **Data feed limits.** Yahoo is unofficial, delayed, caps intraday history at
   about 60 days, and reports zero volume for indices (the package detects this
   and skips volume rules rather than inventing them). NSE endpoints are
   undocumented, rate-limited, and block datacentre IPs — they generally will
   not work from a cloud host. **For anything you trade real money on, use your
   broker's licensed API and `CsvFeed`.**

6. **No execution.** This is analysis only. There is no order placement, no
   broker integration, and no automated trading. That is deliberate: an
   uncalibrated signal engine wired to live orders is how accounts are
   destroyed quickly rather than slowly.

7. **Small-sample statistics.** Any backtest here will produce dozens to a few
   hundred trades. That cannot establish an edge. The metrics say so.

---

## Configuration

Everything that the exchanges control lives in dated YAML you own:

- `config/instruments.yaml` — lot sizes, strike steps and expiry weekdays,
  keyed by effective date
- `config/holidays.yaml` — trading holidays by year, with a `provisional` flag
- `config/costs.yaml` — STT, exchange charges, SEBI fees, stamp duty, GST,
  brokerage and slippage assumptions

Update these when a circular lands. The `verified_on` stamps tell you how old
the numbers are.

## Tests

```bash
pytest -q                       # everything
pytest -q -m "not slow"         # skip the full backtest runs
```

Tests check mathematical identities (put-call parity, the delta relation,
IV round-trips) rather than snapshots, and include an explicit look-ahead
check: truncating the input must not change any earlier indicator value.

---

## Disclaimer

This is analysis software, not investment advice, and it is not a prediction.
It applies published technical and options methods to price data and shows its
reasoning; it has no knowledge of news, order flow or anything else that
actually moves the market. No configuration of it will produce reliably
profitable trading. Backtest it on your own data, paper trade it for months,
and risk only capital you can afford to lose. Derivatives can lose more than
you put in.

MIT licensed.

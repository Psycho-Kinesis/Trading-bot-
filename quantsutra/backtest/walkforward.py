"""Walk-forward validation.

A single backtest over one history tells you almost nothing.  Any ruleset with
enough parameters will fit a fixed window, and the fit will not survive
contact with new data.  Walk-forward is the cheapest defence available: split
the history into consecutive segments, and measure whether performance in each
*out-of-sample* segment resembles the in-sample segment before it.

The number that matters is not the return.  It is the **efficiency ratio** --
out-of-sample performance divided by in-sample performance.  A ratio near 1
means the behaviour generalised.  A ratio near 0, or negative, means the
in-sample result was fitting, and no amount of good-looking equity curve
changes that.

This module does not optimise parameters, deliberately.  Adding a parameter
search here would turn a validation tool into an overfitting machine; if you
want one, run it inside each in-sample window yourself and be honest about the
degrees of freedom you spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import BacktestConfig, Backtester

__all__ = ["WalkForwardResult", "rolling_windows", "walk_forward"]


def rolling_windows(n: int, train: int, test: int, step: int | None = None):
    """Yield ``(train_slice, test_slice)`` index pairs."""
    step = step or test
    start = 0
    while start + train + test <= n:
        yield slice(start, start + train), slice(start + train, start + train + test)
        start += step


@dataclass
class WalkForwardResult:
    folds: list[dict] = field(default_factory=list)
    config: BacktestConfig | None = None

    @property
    def total_oos_trades(self) -> int:
        return sum(f["oos"]["trades"] for f in self.folds)

    def efficiency(self) -> dict:
        """In-sample vs out-of-sample comparison across all folds."""
        usable = [f for f in self.folds
                  if f["is"]["trades"] >= 3 and f["oos"]["trades"] >= 3]
        if not usable:
            return {
                "efficiency_ratio": None, "folds_compared": 0,
                "verdict": "INSUFFICIENT_DATA",
                "note": ("Not enough trades in enough folds to compare in-sample with "
                         "out-of-sample. That is itself informative: a system this "
                         "selective needs far more history before any claim about it "
                         "can be supported."),
            }

        is_exp = np.array([f["is"]["expectancy_r"] for f in usable])
        oos_exp = np.array([f["oos"]["expectancy_r"] for f in usable])
        is_mean = float(np.mean(is_exp))
        oos_mean = float(np.mean(oos_exp))
        ratio = (oos_mean / is_mean) if abs(is_mean) > 1e-9 else None

        oos_positive = int((oos_exp > 0).sum())
        consistency = oos_positive / len(oos_exp)

        if oos_mean <= 0:
            verdict = "FAILED_OUT_OF_SAMPLE"
            note = ("Out-of-sample expectancy is not positive. Whatever the in-sample "
                    "numbers looked like, this did not generalise. Do not trade it.")
        elif ratio is not None and ratio < 0.4:
            verdict = "LIKELY_OVERFIT"
            note = (f"Out-of-sample expectancy is only {ratio:.0%} of in-sample. That gap "
                    f"is the signature of fitting to the training window.")
        elif consistency < 0.5:
            verdict = "INCONSISTENT"
            note = (f"Only {oos_positive}/{len(oos_exp)} out-of-sample folds were "
                    f"profitable. The average hides large fold-to-fold variation, which "
                    f"is what you will actually experience.")
        else:
            verdict = "GENERALISED"
            note = (f"Out-of-sample expectancy held at {ratio:.0%} of in-sample across "
                    f"{len(usable)} folds, positive in {oos_positive} of them. This is "
                    f"the weakest claim worth making -- it survived unseen data. It is "
                    f"not evidence of a durable edge.")

        return {
            "efficiency_ratio": round(ratio, 3) if ratio is not None else None,
            "in_sample_expectancy_r": round(is_mean, 3),
            "out_of_sample_expectancy_r": round(oos_mean, 3),
            "oos_folds_positive": oos_positive,
            "folds_compared": len(usable),
            "consistency": round(consistency, 2),
            "verdict": verdict, "note": note,
        }

    def summary(self) -> str:
        lines = ["=" * 68, "WALK-FORWARD VALIDATION", "=" * 68]
        header = f"{'Fold':>4}  {'In-sample':>28}  {'Out-of-sample':>28}"
        lines.append(header)
        lines.append(f"{'':>4}  {'trades  win%   exp(R)':>28}  {'trades  win%   exp(R)':>28}")
        for fold in self.folds:
            ins, oos = fold["is"], fold["oos"]
            lines.append(
                f"{fold['fold']:>4}  "
                f"{ins['trades']:>6}  {ins['win_rate']*100:>4.0f}%  {ins['expectancy_r']:>+6.2f}"
                f"{'':>10}"
                f"{oos['trades']:>6}  {oos['win_rate']*100:>4.0f}%  {oos['expectancy_r']:>+6.2f}"
            )
        eff = self.efficiency()
        skipped = len(self.folds) - eff["folds_compared"]
        if skipped:
            lines.append(f"\n  {skipped} fold(s) excluded from the comparison: fewer than "
                         f"3 trades on one side, so the expectancy is meaningless there.")
        lines += ["", f"Verdict: {eff['verdict']}", f"  {eff['note']}"]
        if eff.get("efficiency_ratio") is not None:
            lines.append(f"  Efficiency ratio: {eff['efficiency_ratio']:.2f} "
                         f"(out-of-sample expectancy / in-sample expectancy)")
        lines.append(f"  Compared {eff['folds_compared']} of {len(self.folds)} folds, "
                     f"{self.total_oos_trades} out-of-sample trades in total.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"folds": self.folds, "efficiency": self.efficiency()}


def _fold_metrics(result) -> dict:
    metrics = result.report.metrics
    return {
        "trades": metrics["trades"], "win_rate": metrics["win_rate"],
        "expectancy_r": metrics["expectancy_r"], "net_pnl": metrics["net_pnl"],
        "max_drawdown": metrics["max_drawdown"],
        "profit_factor": metrics["profit_factor"],
    }


def walk_forward(
    ohlcv: pd.DataFrame, config: BacktestConfig | None = None,
    train_bars: int = 400, test_bars: int = 150, step_bars: int | None = None,
    progress: bool = False,
) -> WalkForwardResult:
    """Run consecutive in-sample/out-of-sample backtests over the history.

    Each fold needs ``config.warmup_bars`` of context inside its own slice, so
    ``train_bars`` and ``test_bars`` must both exceed the warmup comfortably.
    """
    config = config or BacktestConfig()
    frame = ohlcv.sort_index()
    n = len(frame)
    minimum = config.warmup_bars + 30

    if train_bars < minimum or test_bars < minimum:
        raise ValueError(
            f"train_bars and test_bars must each exceed warmup_bars + 30 = {minimum}; "
            f"got train={train_bars}, test={test_bars}. Either shorten the warmup or "
            f"lengthen the folds."
        )
    if n < train_bars + test_bars:
        raise ValueError(
            f"need at least {train_bars + test_bars} bars for one fold; got {n}"
        )

    result = WalkForwardResult(config=config)
    engine = Backtester(config)

    for i, (train_slice, test_slice) in enumerate(
        rolling_windows(n, train_bars, test_bars, step_bars), start=1
    ):
        train_frame = frame.iloc[train_slice]
        # The out-of-sample slice carries the warmup bars immediately preceding
        # it so its indicators are warm -- those bars are context, not trades.
        oos_start = max(0, test_slice.start - config.warmup_bars)
        test_frame = frame.iloc[oos_start:test_slice.stop]

        if progress:
            print(f"  fold {i}: train {train_frame.index[0].date()}"
                  f"..{train_frame.index[-1].date()}  "
                  f"test {frame.index[test_slice.start].date()}"
                  f"..{frame.index[test_slice.stop - 1].date()}")

        try:
            in_sample = engine.run(train_frame)
            out_sample = engine.run(test_frame)
        except ValueError:
            continue

        result.folds.append({
            "fold": i,
            "train_start": str(train_frame.index[0]), "train_end": str(train_frame.index[-1]),
            "test_start": str(frame.index[test_slice.start]),
            "test_end": str(frame.index[test_slice.stop - 1]),
            "is": _fold_metrics(in_sample), "oos": _fold_metrics(out_sample),
        })
    return result

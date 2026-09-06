"""Shared fixtures."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantsutra.constants import IST

warnings.filterwarnings("ignore", category=UserWarning)


def _frame(closes, seed=0, noise=0.004, volume=True):
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    data = {
        "open": closes * (1 + rng.normal(0, noise / 3, n)),
        "high": closes * (1 + np.abs(rng.normal(0, noise, n))),
        "low": closes * (1 - np.abs(rng.normal(0, noise, n))),
        "close": closes,
    }
    if volume:
        data["volume"] = rng.integers(200_000, 900_000, n)
    frame = pd.DataFrame(data, index=pd.date_range("2023-01-02 15:30", periods=n,
                                                   freq="B", tz=IST))
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


@pytest.fixture
def uptrend():
    rng = np.random.default_rng(1)
    n = 400
    return _frame(20000 * np.exp(np.linspace(0, 0.30, n) + np.cumsum(rng.normal(0, 0.004, n))), seed=1)


@pytest.fixture
def downtrend():
    rng = np.random.default_rng(2)
    n = 400
    return _frame(26000 * np.exp(-np.linspace(0, 0.30, n) + np.cumsum(rng.normal(0, 0.004, n))), seed=2)


@pytest.fixture
def ranging():
    rng = np.random.default_rng(3)
    n, mu, theta = 400, 22000.0, 0.06
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + rng.normal(0, 90)
    return _frame(x, seed=3)


@pytest.fixture
def synthetic():
    from quantsutra.data import generate_index_series
    return generate_index_series(days=600, seed=11)


@pytest.fixture
def option_chain():
    """A synthetic NIFTY chain with a realistic put-skewed smile."""
    import datetime as dt

    from quantsutra.options.chain import chain_from_records
    from quantsutra.options.pricing import bs_price, time_to_expiry

    spot, step, dte = 25000.0, 50, 4
    t = time_to_expiry(dte)
    records = []
    for strike in range(23500, 26550, step):
        strike = float(strike)
        m = (strike - spot) / spot
        iv = 0.13 + 0.9 * m**2 - 0.55 * m
        oi = int(400_000 * np.exp(-((abs(m) * 100) ** 2) / 8)) + 20_000
        records += [
            {"strike": strike, "type": "CE", "ltp": round(bs_price(spot, strike, t, iv, "CE"), 2),
             "oi": oi, "iv": round(iv * 100, 2), "volume": oi // 3, "oi_change": 6000},
            {"strike": strike, "type": "PE", "ltp": round(bs_price(spot, strike, t, iv, "PE"), 2),
             "oi": int(oi * 1.2), "iv": round(iv * 100, 2), "volume": oi // 2, "oi_change": 9000},
        ]
    return chain_from_records("NIFTY", spot, dt.date(2025, 9, 16), records,
                              dte_trading=dte, lot_size=75, strike_step=step)

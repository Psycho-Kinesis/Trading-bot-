"""Option chain analytics: open interest, PCR, max pain, IV surface.

Open interest is the one dataset Indian retail traders have that equity
traders in most markets do not get for free, and it is genuinely informative
*if read correctly*.  Two cautions baked into this module:

* **OI change matters more than OI level.**  A strike with huge OI built up
  weeks ago tells you little; a strike that added 40% of its OI today is where
  positions are being taken right now.

* **OI direction is ambiguous without price.**  Rising OI with rising price is
  a long build-up; rising OI with falling price is a short build-up.  The
  ``classify_oi_buildup`` function encodes the standard 2x2 rather than
  letting callers guess.

Max pain is included because it is widely watched, but see the docstring: its
predictive power is weak and it is treated as a minor input, not a thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pricing import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, greeks, implied_volatility, moneyness

__all__ = [
    "OptionChain",
    "chain_from_records",
    "classify_oi_buildup",
    "iv_rank",
    "iv_skew",
    "max_pain",
    "put_call_ratio",
    "support_resistance_from_oi",
]


def classify_oi_buildup(price_change: float, oi_change: float,
                        threshold_pct: float = 0.0) -> str:
    """The standard price/OI 2x2.

    ============  ==============  ====================================
    Price         Open interest   Interpretation
    ============  ==============  ====================================
    Up            Up              LONG_BUILDUP -- new longs, bullish
    Down          Up              SHORT_BUILDUP -- new shorts, bearish
    Up            Down            SHORT_COVERING -- shorts exiting, bullish but weaker
    Down          Down            LONG_UNWINDING -- longs exiting, bearish but weaker
    ============  ==============  ====================================

    Build-ups signal conviction; unwinding signals exit.  A rally on short
    covering is structurally weaker than a rally on long build-up, and that
    distinction is what the signal engine uses to grade follow-through.
    """
    if not np.isfinite(price_change) or not np.isfinite(oi_change):
        return "UNKNOWN"
    if abs(price_change) <= threshold_pct:
        return "NEUTRAL"
    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    if price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    if price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    if price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    return "NEUTRAL"


@dataclass
class OptionChain:
    """A snapshot of one expiry's option chain.

    ``data`` is a frame indexed by strike with columns:
    ``ce_ltp, ce_oi, ce_oi_change, ce_volume, ce_iv, pe_ltp, pe_oi,
    pe_oi_change, pe_volume, pe_iv`` (missing columns are tolerated).
    """

    symbol: str
    spot: float
    expiry: object
    data: pd.DataFrame
    dte_trading: int = 1
    lot_size: int = 1
    strike_step: int = 50
    rate: float = DEFAULT_RATE
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
    timestamp: object = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = self.data.sort_index()
        for col in ("ce_ltp", "ce_oi", "ce_oi_change", "ce_volume", "ce_iv",
                    "pe_ltp", "pe_oi", "pe_oi_change", "pe_volume", "pe_iv"):
            if col not in self.data.columns:
                self.data[col] = np.nan

    # -- basic geometry --------------------------------------------------
    @property
    def atm_strike(self) -> float:
        strikes = self.data.index.to_numpy(dtype=float)
        if not len(strikes):
            return float("nan")
        return float(strikes[np.argmin(np.abs(strikes - self.spot))])

    @property
    def t(self) -> float:
        from .pricing import time_to_expiry
        return time_to_expiry(max(self.dte_trading, 0.25))

    def strikes_around_atm(self, n: int = 5) -> pd.DataFrame:
        atm = self.atm_strike
        lo, hi = atm - n * self.strike_step, atm + n * self.strike_step
        return self.data.loc[(self.data.index >= lo) & (self.data.index <= hi)]

    # -- derived analytics ------------------------------------------------
    def compute_ivs(self, overwrite: bool = False) -> pd.DataFrame:
        """Fill missing IVs by inverting BSM on the traded price."""
        out = self.data.copy()
        for side, col_ltp, col_iv in (("CE", "ce_ltp", "ce_iv"), ("PE", "pe_ltp", "pe_iv")):
            need = out[col_iv].isna() | overwrite
            for strike in out.index[need]:
                ltp = out.at[strike, col_ltp]
                if not np.isfinite(ltp) or ltp <= 0:
                    continue
                out.at[strike, col_iv] = implied_volatility(
                    float(ltp), self.spot, float(strike), self.t, side,
                    self.rate, self.dividend_yield,
                ) * 100
        self.data = out
        return out

    def greeks_table(self, n: int = 5) -> pd.DataFrame:
        """Greeks for the strikes around ATM, using each strike's own IV."""
        rows = []
        for strike, row in self.strikes_around_atm(n).iterrows():
            for side, iv_col in (("CE", "ce_iv"), ("PE", "pe_iv")):
                iv = row.get(iv_col)
                if not np.isfinite(iv) or iv <= 0:
                    continue
                g = greeks(self.spot, float(strike), self.t, float(iv) / 100, side,
                           self.rate, self.dividend_yield)
                rows.append({
                    "strike": float(strike), "type": side, "iv": round(float(iv), 2),
                    "moneyness": moneyness(self.spot, float(strike), side),
                    **g.to_dict(),
                })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Everything the signal engine reads off the chain, in one dict."""
        pcr = put_call_ratio(self)
        skew = iv_skew(self)
        walls = support_resistance_from_oi(self)
        mp = max_pain(self)
        atm = self.atm_strike
        atm_row = self.data.loc[atm] if atm in self.data.index else None

        atm_iv = np.nan
        if atm_row is not None:
            ivs = [atm_row.get("ce_iv"), atm_row.get("pe_iv")]
            ivs = [float(x) for x in ivs if x is not None and np.isfinite(x)]
            atm_iv = float(np.mean(ivs)) if ivs else np.nan

        return {
            "symbol": self.symbol, "spot": round(self.spot, 2), "expiry": str(self.expiry),
            "dte_trading": self.dte_trading, "atm_strike": atm,
            "atm_iv": round(atm_iv, 2) if np.isfinite(atm_iv) else None,
            **pcr, **skew, **walls,
            "max_pain": mp["max_pain"],
            "max_pain_distance_pct": mp["distance_pct"],
            "total_ce_oi": int(np.nansum(self.data["ce_oi"].to_numpy(float))) if self.data["ce_oi"].notna().any() else None,
            "total_pe_oi": int(np.nansum(self.data["pe_oi"].to_numpy(float))) if self.data["pe_oi"].notna().any() else None,
            "warnings": self.warnings,
        }


def chain_from_records(
    symbol: str, spot: float, expiry, records: list[dict], dte_trading: int = 1,
    lot_size: int = 1, strike_step: int = 50,
) -> OptionChain:
    """Build a chain from a list of ``{strike, type, ltp, oi, ...}`` dicts.

    Accepts the shape returned by the NSE option-chain endpoint as well as
    broker APIs, so feeds can be swapped without touching the analytics.
    """
    rows: dict[float, dict] = {}
    for rec in records:
        try:
            strike = float(rec["strike"] if "strike" in rec else rec["strikePrice"])
        except (KeyError, TypeError, ValueError):
            continue
        side = str(rec.get("type") or rec.get("option_type") or "").upper()
        if side in ("C", "CALL"):
            side = "CE"
        elif side in ("P", "PUT"):
            side = "PE"
        if side not in ("CE", "PE"):
            continue
        prefix = side.lower()
        row = rows.setdefault(strike, {})
        row[f"{prefix}_ltp"] = _num(rec.get("ltp", rec.get("lastPrice")))
        row[f"{prefix}_oi"] = _num(rec.get("oi", rec.get("openInterest")))
        row[f"{prefix}_oi_change"] = _num(rec.get("oi_change", rec.get("changeinOpenInterest")))
        row[f"{prefix}_volume"] = _num(rec.get("volume", rec.get("totalTradedVolume")))
        row[f"{prefix}_iv"] = _num(rec.get("iv", rec.get("impliedVolatility")))
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "strike"
    return OptionChain(symbol=symbol, spot=spot, expiry=expiry, data=frame,
                       dte_trading=dte_trading, lot_size=lot_size, strike_step=strike_step)


def _num(value) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def put_call_ratio(chain: OptionChain, band: int = 10) -> dict:
    """PCR by open interest and by volume, overall and near the money.

    Read counter-intuitively: a *high* PCR means heavy put writing, which is
    usually bullish positioning, and vice versa.  Extremes matter far more than
    the level -- PCR above ~1.5 or below ~0.6 marks crowded positioning that
    tends to unwind.
    """
    d = chain.data
    near = chain.strikes_around_atm(band)

    def _ratio(frame: pd.DataFrame, pe_col: str, ce_col: str) -> float:
        pe = float(np.nansum(frame[pe_col].to_numpy(float)))
        ce = float(np.nansum(frame[ce_col].to_numpy(float)))
        return pe / ce if ce > 0 else float("nan")

    pcr_oi = _ratio(d, "pe_oi", "ce_oi")
    pcr_vol = _ratio(d, "pe_volume", "ce_volume")
    pcr_near = _ratio(near, "pe_oi", "ce_oi")

    if not np.isfinite(pcr_oi):
        reading = "UNKNOWN"
    elif pcr_oi > 1.5:
        reading = "HEAVY_PUT_WRITING_BULLISH_BUT_CROWDED"
    elif pcr_oi > 1.0:
        reading = "PUT_WRITING_BULLISH"
    elif pcr_oi > 0.7:
        reading = "BALANCED"
    elif pcr_oi > 0.5:
        reading = "CALL_WRITING_BEARISH"
    else:
        reading = "HEAVY_CALL_WRITING_BEARISH_BUT_CROWDED"

    return {
        "pcr_oi": round(pcr_oi, 3) if np.isfinite(pcr_oi) else None,
        "pcr_volume": round(pcr_vol, 3) if np.isfinite(pcr_vol) else None,
        "pcr_near_atm": round(pcr_near, 3) if np.isfinite(pcr_near) else None,
        "pcr_reading": reading,
    }


def max_pain(chain: OptionChain) -> dict:
    """The strike at which total option-writer payout is minimised.

    Treat this as weak evidence.  The academic result is that max pain has
    little predictive power outside the last few hours of expiry, and even
    then the effect is small relative to a real directional move.  It is
    reported because a pin near max pain does happen on quiet expiry days, and
    the signal engine gives it a small weight only when DTE is 0 or 1.
    """
    d = chain.data
    strikes = d.index.to_numpy(dtype=float)
    ce_oi = np.nan_to_num(d["ce_oi"].to_numpy(float))
    pe_oi = np.nan_to_num(d["pe_oi"].to_numpy(float))
    if not len(strikes) or (ce_oi.sum() + pe_oi.sum()) <= 0:
        return {"max_pain": None, "distance_pct": None, "pain_curve": []}

    pains = []
    for settle in strikes:
        call_pain = float(np.sum(np.maximum(settle - strikes, 0) * ce_oi))
        put_pain = float(np.sum(np.maximum(strikes - settle, 0) * pe_oi))
        pains.append(call_pain + put_pain)
    idx = int(np.argmin(pains))
    mp = float(strikes[idx])
    return {
        "max_pain": mp,
        "distance_pct": round(100 * (mp - chain.spot) / chain.spot, 3),
        "pain_curve": [{"strike": float(s), "pain": float(p)} for s, p in zip(strikes, pains, strict=True)],
    }


def support_resistance_from_oi(chain: OptionChain, top_n: int = 3) -> dict:
    """Strikes with the heaviest call/put OI act as resistance/support.

    Mechanically: call writers at a strike hedge by selling into rallies toward
    it, put writers buy dips toward theirs.  The effect is real but it *fails
    on trend days*, which is exactly when it matters most -- so these are
    reported as levels, never as a reason to fade a strong move.
    """
    d = chain.data
    out: dict = {}

    ce = d["ce_oi"].dropna()
    if len(ce):
        top_ce = ce.nlargest(top_n)
        out["call_oi_resistance"] = [
            {"strike": float(k), "oi": int(v)} for k, v in top_ce.items()
        ]
        out["max_call_oi_strike"] = float(top_ce.index[0])
    pe = d["pe_oi"].dropna()
    if len(pe):
        top_pe = pe.nlargest(top_n)
        out["put_oi_support"] = [{"strike": float(k), "oi": int(v)} for k, v in top_pe.items()]
        out["max_put_oi_strike"] = float(top_pe.index[0])

    # Where new positions are being added *today*.
    ce_chg = d["ce_oi_change"].dropna()
    pe_chg = d["pe_oi_change"].dropna()
    if len(ce_chg):
        out["biggest_ce_oi_addition"] = float(ce_chg.idxmax())
    if len(pe_chg):
        out["biggest_pe_oi_addition"] = float(pe_chg.idxmax())
    if len(ce_chg) and len(pe_chg):
        ce_add = float(ce_chg.clip(lower=0).sum())
        pe_add = float(pe_chg.clip(lower=0).sum())
        total = ce_add + pe_add
        out["oi_flow_bias"] = (
            "BULLISH_PUT_WRITING" if total > 0 and pe_add / total > 0.60 else
            "BEARISH_CALL_WRITING" if total > 0 and ce_add / total > 0.60 else
            "BALANCED"
        )
    return out


def iv_skew(chain: OptionChain, wing_strikes: int = 5) -> dict:
    """ATM IV plus the put-call skew across the wings.

    Indian index options normally carry a *put* skew (downside puts priced
    above equidistant calls) because the tail risk is asymmetric.  Skew
    flattening or inverting is a meaningful sentiment shift: when calls start
    pricing above puts, the market is paying up for upside, which historically
    marks late-stage rallies.
    """
    d = chain.data
    atm = chain.atm_strike
    if not np.isfinite(atm):
        return {"atm_iv": None, "iv_skew": None, "skew_reading": "UNKNOWN"}

    step = chain.strike_step
    wing = wing_strikes * step
    otm_put = atm - wing
    otm_call = atm + wing

    def _iv(strike: float, col: str) -> float:
        if strike in d.index:
            val = d.at[strike, col]
            return float(val) if val is not None and np.isfinite(val) else np.nan
        return np.nan

    put_iv = _iv(otm_put, "pe_iv")
    call_iv = _iv(otm_call, "ce_iv")
    atm_ce = _iv(atm, "ce_iv")
    atm_pe = _iv(atm, "pe_iv")
    atm_iv = float(np.nanmean([atm_ce, atm_pe]))

    skew = put_iv - call_iv if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan
    if not np.isfinite(skew):
        reading = "UNKNOWN"
    elif skew > 3:
        reading = "STEEP_PUT_SKEW_FEAR_BID"
    elif skew > 0.5:
        reading = "NORMAL_PUT_SKEW"
    elif skew > -0.5:
        reading = "FLAT_SKEW"
    else:
        reading = "CALL_SKEW_UPSIDE_CHASED"

    return {
        "atm_iv": round(atm_iv, 2) if np.isfinite(atm_iv) else None,
        "otm_put_iv": round(put_iv, 2) if np.isfinite(put_iv) else None,
        "otm_call_iv": round(call_iv, 2) if np.isfinite(call_iv) else None,
        "iv_skew": round(float(skew), 2) if np.isfinite(skew) else None,
        "skew_reading": reading,
    }


def iv_rank(current_iv: float, history: pd.Series | np.ndarray) -> dict:
    """IV rank and IV percentile against a trailing history.

    IV *rank* places IV between its period low and high; IV *percentile* is the
    fraction of days it traded below the current level.  Percentile is the more
    robust of the two -- rank is distorted by a single spike (a COVID-March
    print keeps rank near zero for a year afterwards).
    """
    hist = pd.Series(history).dropna().astype(float)
    if len(hist) < 20 or not np.isfinite(current_iv):
        return {"iv_rank": None, "iv_percentile": None, "iv_reading": "INSUFFICIENT_HISTORY"}
    lo, hi = float(hist.min()), float(hist.max())
    rank = 100 * (current_iv - lo) / (hi - lo) if hi > lo else float("nan")
    pctile = float((hist < current_iv).mean() * 100)
    if pctile > 80:
        reading = "IV_EXPENSIVE_FAVOUR_SELLING"
    elif pctile > 55:
        reading = "IV_ABOVE_AVERAGE"
    elif pctile > 25:
        reading = "IV_AVERAGE"
    else:
        reading = "IV_CHEAP_FAVOUR_BUYING"
    return {
        "iv_rank": round(rank, 1) if np.isfinite(rank) else None,
        "iv_percentile": round(pctile, 1),
        "iv_reading": reading,
    }

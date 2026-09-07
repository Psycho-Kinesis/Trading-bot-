"""Trade journal.

Every signal the engine produces is logged to SQLite before anything is acted
on, along with the full reasoning.  This exists for one reason: to make the
system's record checkable after the fact.

Without a journal, memory quietly rewrites history -- you remember the calls
that worked and forget the ones that did not, and the system appears better
than it is.  The journal makes that impossible: :meth:`hit_rate` reports the
realised outcome of every logged signal, including the ones you would rather
not count.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

__all__ = ["Journal"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT,
    direction TEXT,
    action TEXT,
    confidence REAL,
    raw_score REAL,
    spot REAL,
    entry REAL,
    stop_loss REAL,
    target REAL,
    regime TEXT,
    actionable INTEGER,
    vetoes TEXT,
    reasons TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    signal_id INTEGER PRIMARY KEY,
    resolved_at TEXT,
    exit_price REAL,
    outcome TEXT,          -- TARGET | STOP | TIME | UNRESOLVED
    pnl REAL,
    r_multiple REAL,
    notes TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals (id)
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals (symbol, created_at);
"""


class Journal:
    def __init__(self, path: str | Path = "quantsutra_journal.sqlite"):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- writing -----------------------------------------------------------
    def log_signal(self, signal: dict, payload: dict | None = None) -> int:
        """Record a signal, actionable or not.

        No-trade signals are logged too -- a system that only records its
        opinions when it has one cannot be evaluated honestly.
        """
        cursor = self.connection.execute(
            """INSERT INTO signals (created_at, symbol, timeframe, direction, action,
                                    confidence, raw_score, spot, entry, stop_loss, target,
                                    regime, actionable, vetoes, reasons, payload)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal.get("timestamp") or dt.datetime.now().isoformat(),
                signal.get("symbol"), signal.get("timeframe"),
                signal.get("direction"), signal.get("action"),
                signal.get("confidence"), signal.get("raw_score"), signal.get("spot"),
                signal.get("entry"), signal.get("stop_loss"),
                (signal.get("targets") or [None])[0],
                (signal.get("regime") or {}).get("regime"),
                1 if signal.get("actionable") else 0,
                json.dumps(signal.get("vetoes") or []),
                json.dumps(signal.get("reasons") or []),
                json.dumps(payload or signal, default=str),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_outcome(self, signal_id: int, exit_price: float, outcome: str,
                       pnl: float | None = None, notes: str = "") -> None:
        row = self.connection.execute(
            "SELECT entry, stop_loss FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()
        r_multiple = None
        if row and row["entry"] and row["stop_loss"]:
            risk = abs(row["entry"] - row["stop_loss"])
            if risk > 0:
                r_multiple = (exit_price - row["entry"]) / risk
        self.connection.execute(
            """INSERT OR REPLACE INTO outcomes
               (signal_id, resolved_at, exit_price, outcome, pnl, r_multiple, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (signal_id, dt.datetime.now().isoformat(), exit_price, outcome, pnl,
             r_multiple, notes),
        )
        self.connection.commit()

    # -- reading -----------------------------------------------------------
    def recent(self, symbol: str | None = None, limit: int = 20) -> list[dict]:
        query = "SELECT * FROM signals"
        params: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            params = (symbol.upper(),)
        query += " ORDER BY id DESC LIMIT ?"
        rows = self.connection.execute(query, params + (limit,)).fetchall()
        return [dict(row) for row in rows]

    def last_signal(self, symbol: str) -> dict | None:
        rows = self.recent(symbol, 1)
        return rows[0] if rows else None

    def hit_rate(self, symbol: str | None = None) -> dict:
        """Realised outcome of every *resolved* signal.

        Unresolved signals are counted and reported separately rather than
        dropped, because silently excluding them is how a hit rate gets
        flattered.
        """
        query = """SELECT s.action, s.confidence, o.outcome, o.r_multiple
                   FROM signals s LEFT JOIN outcomes o ON s.id = o.signal_id
                   WHERE s.actionable = 1"""
        params: tuple = ()
        if symbol:
            query += " AND s.symbol = ?"
            params = (symbol.upper(),)
        rows = self.connection.execute(query, params).fetchall()

        resolved = [r for r in rows if r["outcome"]]
        unresolved = len(rows) - len(resolved)
        if not resolved:
            return {
                "actionable_signals": len(rows), "resolved": 0,
                "unresolved": unresolved, "hit_rate": None,
                "note": ("No resolved outcomes yet. Record them with "
                         "record_outcome() -- an unaudited signal log proves nothing."),
            }

        wins = sum(1 for r in resolved if r["outcome"] == "TARGET")
        r_values = [r["r_multiple"] for r in resolved if r["r_multiple"] is not None]

        from ..backtest.metrics import win_rate_confidence_interval
        low, high = win_rate_confidence_interval(wins, len(resolved))

        note = ""
        if len(resolved) < 30:
            note = (f"Only {len(resolved)} resolved signals. The 95% interval on this hit "
                    f"rate is {low:.0%}-{high:.0%}, which is too wide to mean anything.")
        elif low < 0.5 < high:
            note = ("The confidence interval straddles 50%: this record is consistent "
                    "with having no edge.")

        return {
            "actionable_signals": len(rows), "resolved": len(resolved),
            "unresolved": unresolved,
            "hit_rate": round(wins / len(resolved), 3),
            "hit_rate_ci": [round(low, 3), round(high, 3)],
            "avg_r": round(sum(r_values) / len(r_values), 3) if r_values else None,
            "note": note,
        }

    def stats_by_regime(self) -> list[dict]:
        """Where the system actually works -- and where it does not."""
        rows = self.connection.execute(
            """SELECT s.regime, COUNT(*) AS n,
                      AVG(CASE WHEN o.outcome = 'TARGET' THEN 1.0 ELSE 0.0 END) AS hit,
                      AVG(o.r_multiple) AS avg_r
               FROM signals s JOIN outcomes o ON s.id = o.signal_id
               WHERE s.actionable = 1 GROUP BY s.regime ORDER BY n DESC"""
        ).fetchall()
        return [
            {"regime": r["regime"], "signals": r["n"],
             "hit_rate": round(r["hit"], 3) if r["hit"] is not None else None,
             "avg_r": round(r["avg_r"], 3) if r["avg_r"] is not None else None}
            for r in rows
        ]

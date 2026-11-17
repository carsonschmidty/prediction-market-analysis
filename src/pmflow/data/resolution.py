"""Derive a conservative resolution outcome for Polymarket markets.

Gamma's `outcomePrices` on a closed market is the settlement payout vector when the market
resolved cleanly (e.g. ["1","0"]). We only call a market `resolved` when exactly one outcome
pays >= 0.99 and every other pays <= 0.01, and the UMA status (when present) is 'resolved'.
Everything else is labelled explicitly instead of guessed:

  ambiguous_5050         all outcomes ~0.5 (UMA 'unknown/50-50' resolution)
  cancelled_or_void      all outcome prices 0 (never settled / voided / legacy)
  unresolved             not closed, UMA still proposed/disputed, or prices not at a corner
  multi_outcome_unclear  >2 outcomes with a non-degenerate payout vector
"""
from __future__ import annotations

from .common import jload

RESOLVED_UMA = {"resolved", "settled"}


def derive_resolution(m: dict) -> dict:
    out = {"resolution_status": "unresolved", "winning_outcome_index": None,
           "winning_outcome": None, "resolved_yes": None}
    outcomes = jload(m.get("outcomes")) or []
    prices_raw = jload(m.get("outcomePrices")) or []
    try:
        prices = [float(p) for p in prices_raw]
    except (TypeError, ValueError):
        prices = []
    if not m.get("closed") or not prices:
        return out
    uma = (m.get("umaResolutionStatus") or "").lower()
    if uma and uma not in RESOLVED_UMA:
        # proposed / disputed / challenged etc.: not final
        return out
    if all(p == 0 for p in prices):
        out["resolution_status"] = "cancelled_or_void"
        return out
    if len(prices) >= 2 and all(abs(p - 1.0 / len(prices)) < 0.02 for p in prices):
        out["resolution_status"] = "ambiguous_5050"
        return out
    winners = [i for i, p in enumerate(prices) if p >= 0.99]
    losers = [i for i, p in enumerate(prices) if p <= 0.01]
    if len(winners) == 1 and len(losers) == len(prices) - 1:
        i = winners[0]
        out["resolution_status"] = "resolved"
        out["winning_outcome_index"] = i
        out["winning_outcome"] = outcomes[i] if i < len(outcomes) else None
        names = [str(o).strip().lower() for o in outcomes]
        if names == ["yes", "no"]:
            out["resolved_yes"] = (i == 0)
        return out
    if len(prices) > 2:
        out["resolution_status"] = "multi_outcome_unclear"
    return out

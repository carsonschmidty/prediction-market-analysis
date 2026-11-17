"""Reported-case loader for the labeling role.

Reads ``artifacts/labeling/reported_cases.csv`` (publicly reported suspected-informed-trading
cases plus negative/contrast cases) into typed records, and - if the data role's
``pm_markets`` table exists in Postgres - tries to match each case to a market id.

The CSV is the source of truth; nothing here invents values. Blank cells become ``None``.
Timestamps carry an explicit precision; a day-precision value is NOT a midnight timestamp
and callers must not treat it as one (use ``news_window()``).

Vindication (``outcome_vindicated``) is a LABEL derived from the resolved outcome. It must
never be used as a model feature.

CLI:  python -m pmflow.labeling.cases [--csv PATH] [--match] [--write-matches PATH]
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "artifacts" / "labeling" / "reported_cases.csv"
DEFAULT_DSN = "postgresql://pmflow:pmflow@localhost:5432/pmflow"

PRECISIONS = {"minute", "hour", "day", "per_event", "not_applicable", "unknown", ""}
GRADES = {"A", "B", "C"}


def _none(v: Optional[str]) -> Optional[str]:
    v = (v or "").strip()
    return v or None


def _split(v: Optional[str]) -> list[str]:
    """Split multi-valued cells (``;`` or `` | `` separated)."""
    v = _none(v)
    if not v:
        return []
    sep = " | " if " | " in v else ";"
    return [p.strip() for p in v.split(sep) if p.strip()]


def parse_ts(v: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamps. Date-only (YYYY-MM-DD) and month-only (YYYY-MM) values are
    returned as the start of that period in UTC; check the precision field before use."""
    v = _none(v)
    if not v:
        return None
    v = v.split(" (")[0].strip()  # tolerate trailing "(note)"
    try:
        if len(v) == 7:  # YYYY-MM
            return datetime.strptime(v, "%Y-%m").replace(tzinfo=timezone.utc)
        if len(v) == 10:  # YYYY-MM-DD
            return datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _bool(v: Optional[str]) -> Optional[bool]:
    v = (_none(v) or "").lower()
    if v in {"true", "yes", "1"}:
        return True
    if v in {"false", "no", "0"}:
        return False
    return None  # 'pending', blank, unknown


@dataclass
class ReportedCase:
    case_id: str
    platform: str
    case_type: str
    market_title_reported: str
    market_slugs: list[str]
    condition_ids: list[str]
    gamma_market_ids: list[int]
    event_slug: Optional[str]
    match_source: Optional[str]
    match_confidence: Optional[str]
    category: Optional[str]
    geo_subtypes: list[str]
    in_scope: Optional[bool]
    event_description: Optional[str]
    resolution: Optional[str]
    public_news_ts: Optional[datetime]
    public_news_ts_upper_bound: Optional[datetime]
    ts_precision: str
    ts_confidence: Optional[str]
    ts_source_url: Optional[str]
    ts_note: Optional[str]
    official_confirmation_ts: Optional[datetime]
    flow_window_start: Optional[datetime]
    flow_window_end: Optional[datetime]
    flow_window_note: Optional[str]
    reported_size_usd: Optional[str]          # free text: sources disagree / ranges
    reported_profit_usd: Optional[str]
    reported_entry_price: Optional[str]
    account_pattern: Optional[str]
    opaque_ids_published: list[str]
    official_action: Optional[str]
    evidence_grade: Optional[str]
    outcome_vindicated: Optional[bool]        # LABEL ONLY - never a feature
    source_urls: list[str]
    source_access: Optional[str]
    uncertainty_notes: Optional[str]
    raw: dict = field(default_factory=dict, repr=False)
    # filled by match_to_markets()
    pg_market_ids: list[int] = field(default_factory=list)
    pg_match_method: Optional[str] = None

    @property
    def is_negative(self) -> bool:
        return self.case_type.startswith("negative") or self.outcome_vindicated is False

    @property
    def has_fine_timestamp(self) -> bool:
        return self.public_news_ts is not None and self.ts_precision in {"minute", "hour"}

    def news_window(self) -> Optional[tuple[datetime, datetime]]:
        """[earliest, latest] interval in which the information became public.

        minute -> +/-5 min, hour -> +/-1 h. day -> the CSV date is the *local* calendar date
        in the reporting jurisdiction (often US ET), so the window runs from 00:00Z of that
        date to 12:00Z of the next day, clipped by the upper bound if one is given.
        A case with only an upper bound returns (bound - 1 day, bound)."""
        t, ub, p = self.public_news_ts, self.public_news_ts_upper_bound, self.ts_precision
        if t is not None:
            pad = {"minute": timedelta(minutes=5), "hour": timedelta(hours=1)}.get(p)
            if pad is not None:
                return t - pad, t + pad
            if p == "day":
                end = t + timedelta(days=1, hours=12)
                if ub is not None:
                    end = min(end, ub)
                return t, end
        if ub is not None:
            return ub - timedelta(days=1), ub
        return None


def load_cases(path: os.PathLike | str = DEFAULT_CSV) -> list[ReportedCase]:
    out: list[ReportedCase] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            prec = (r.get("ts_precision") or "").strip()
            if prec not in PRECISIONS:
                raise ValueError(f"{r['case_id']}: unknown ts_precision {prec!r}")
            grade = _none(r.get("evidence_grade"))
            if grade and grade not in GRADES:
                raise ValueError(f"{r['case_id']}: unknown evidence_grade {grade!r}")
            out.append(ReportedCase(
                case_id=r["case_id"].strip(),
                platform=r["platform"].strip(),
                case_type=r["case_type"].strip(),
                market_title_reported=r["market_title_reported"].strip(),
                market_slugs=_split(r.get("market_slug")),
                condition_ids=_split(r.get("condition_id")),
                gamma_market_ids=[int(x) for x in _split(r.get("gamma_market_id"))],
                event_slug=_none(r.get("event_slug")),
                match_source=_none(r.get("match_source")),
                match_confidence=_none(r.get("match_confidence")),
                category=_none(r.get("category")),
                geo_subtypes=_split(r.get("geo_subtype")),
                in_scope=_bool(r.get("in_scope")),
                event_description=_none(r.get("event_description")),
                resolution=_none(r.get("resolution")),
                public_news_ts=parse_ts(r.get("public_news_ts_utc")),
                public_news_ts_upper_bound=parse_ts(r.get("public_news_ts_upper_bound_utc")),
                ts_precision=prec,
                ts_confidence=_none(r.get("ts_confidence")),
                ts_source_url=_none(r.get("ts_source_url")),
                ts_note=_none(r.get("ts_note")),
                official_confirmation_ts=parse_ts(r.get("official_confirmation_ts_utc")),
                flow_window_start=parse_ts(r.get("flow_window_start_utc")),
                flow_window_end=parse_ts(r.get("flow_window_end_utc")),
                flow_window_note=_none(r.get("flow_window_note")),
                reported_size_usd=_none(r.get("reported_size_usd")),
                reported_profit_usd=_none(r.get("reported_profit_usd")),
                reported_entry_price=_none(r.get("reported_entry_price")),
                account_pattern=_none(r.get("account_pattern")),
                opaque_ids_published=_split(r.get("opaque_ids_published")),
                official_action=_none(r.get("official_action")),
                evidence_grade=grade,
                outcome_vindicated=_bool(r.get("outcome_vindicated")),
                source_urls=_split(r.get("source_urls")),
                source_access=_none(r.get("source_access")),
                uncertainty_notes=_none(r.get("uncertainty_notes")),
                raw=dict(r),
            ))
    ids = [c.case_id for c in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in CSV")
    return out


# ---------------------------------------------------------------- Postgres matching
def match_to_markets(cases: Iterable[ReportedCase], dsn: Optional[str] = None) -> dict:
    """Match cases to ``pm_markets.market_id`` by condition_id, then slug, then gamma id.

    Returns a status dict. If Postgres or the table is unavailable, nothing is changed and
    the reason is reported (callers should log it; do not fall back to guessing)."""
    cases = list(cases)
    dsn = dsn or os.environ.get("PMFLOW_DSN", DEFAULT_DSN)
    try:
        import psycopg  # type: ignore
    except ImportError as e:  # pragma: no cover
        return {"status": "no_driver", "detail": str(e), "matched": 0, "total": len(cases)}
    try:
        conn = psycopg.connect(dsn, connect_timeout=5)
    except Exception as e:  # noqa: BLE001
        return {"status": "db_unreachable", "detail": str(e).splitlines()[0], "matched": 0, "total": len(cases)}
    matched = 0
    with conn:
        cur = conn.cursor()
        cur.execute("select to_regclass('public.pm_markets')")
        if cur.fetchone()[0] is None:
            return {"status": "no_pm_markets_table", "matched": 0, "total": len(cases)}
        cur.execute("select count(*) from pm_markets")
        n_markets = cur.fetchone()[0]
        for c in cases:
            c.pg_market_ids, c.pg_match_method = [], None
            for method, col, vals in (
                ("condition_id", "lower(condition_id)", [v.lower() for v in c.condition_ids]),
                ("slug", "slug", c.market_slugs),
                ("gamma_market_id", "market_id", c.gamma_market_ids),
            ):
                if not vals:
                    continue
                cur.execute(f"select market_id from pm_markets where {col} = any(%s)", (vals,))
                ids = [row[0] for row in cur.fetchall()]
                if ids:
                    c.pg_market_ids, c.pg_match_method = sorted(set(ids)), method
                    matched += 1
                    break
    return {"status": "ok", "pm_markets_rows": n_markets, "matched": matched, "total": len(cases)}


def summarize(cases: list[ReportedCase]) -> dict:
    from collections import Counter
    return {
        "n": len(cases),
        "by_platform": dict(Counter(c.platform for c in cases)),
        "by_category": dict(Counter(c.category for c in cases)),
        "by_grade": dict(Counter(c.evidence_grade for c in cases)),
        "by_case_type": dict(Counter(c.case_type for c in cases)),
        "in_scope": sum(1 for c in cases if c.in_scope),
        "minute_or_hour_ts": sum(1 for c in cases if c.has_fine_timestamp),
        "negatives": sum(1 for c in cases if c.is_negative),
        "with_gamma_match": sum(1 for c in cases if c.condition_ids),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--match", action="store_true", help="try matching to pm_markets in Postgres")
    ap.add_argument("--write-matches", help="write case_id,pg_market_ids,method CSV here")
    a = ap.parse_args(argv)
    cases = load_cases(a.csv)
    for k, v in summarize(cases).items():
        print(f"{k}: {v}")
    if a.match or a.write_matches:
        st = match_to_markets(cases)
        print("postgres match:", st)
        if a.write_matches and st.get("status") == "ok":
            with open(a.write_matches, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["case_id", "pg_market_ids", "pg_match_method"])
                for c in cases:
                    w.writerow([c.case_id, ";".join(map(str, c.pg_market_ids)), c.pg_match_method or ""])


if __name__ == "__main__":
    main()

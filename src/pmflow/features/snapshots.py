"""Materialise window features for the evaluation grid into ``feat_snapshot_features``.

One row per ``eval_snapshots.snapshot_id`` (wide: ``<feature>_<L>h`` for L in 1/6/24/72 plus
cutoff-level columns). Every value uses only trades with ts < cutoff_ts (see core.py).

Default snapshot set: ``grid_24h AND is_live AND split <> 'purged_test'`` (178k rows, ~200 MB as
float4). ``--set live`` adds the 6-hourly live grid (~713k rows, ~0.8 GB: only if disk allows).

Resumable: finished markets are recorded in ``feat_snapshot_build_state``; re-running skips
them. ``--rebuild`` drops both tables first.

CLI: python -m pmflow.features.snapshots [--set grid24|live] [--workers 2] [--rebuild] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .core import LOOKBACKS_H, compute_features_batch, to_wide
from .db import connect

TABLE = "feat_snapshot_features"
STATE = "feat_snapshot_build_state"
DROP = {"px_start", "px_end", "log_gv", "yes_share"}   # redundant with kept columns
SETS = {
    "grid24": "grid_24h and is_live and split <> 'purged_test'",
    "live": "is_live and split <> 'purged_test'",
}
TEXT = {"category", "geo_subtype"}
BOOL = {"wallet_age_censored", "market_history_truncated", "in_scope"}


def _wide(conn, req: pd.DataFrame) -> pd.DataFrame:
    long = compute_features_batch(req, conn=conn)
    long = long.drop(columns=[c for c in long.columns if c in DROP])
    w = to_wide(long, key=("snapshot_id",))
    return w


def _columns(conn, where: str) -> list[tuple[str, str]]:
    req = pd.read_sql_query(
        f"select snapshot_id, market_id, cutoff_ts from eval_snapshots where market_id = "
        f"(select market_id from eval_snapshots where {where} group by 1 order by count(*) desc limit 1)"
        f" and {where} limit 3", conn)
    w = _wide(conn, req)
    cols = [("snapshot_id", "bigint primary key"), ("market_id", "bigint"), ("cutoff_ts", "timestamptz")]
    for c in w.columns:
        if c in ("snapshot_id", "market_id", "cutoff_ts"):
            continue
        cols.append((c, "text" if c in TEXT else "boolean" if c in BOOL else "real"))
    return cols


def _ensure(conn, where: str, rebuild: bool):
    with conn.cursor() as cur:
        if rebuild:
            cur.execute(f"drop table if exists {TABLE}")
            cur.execute(f"drop table if exists {STATE}")
        cur.execute(f"create table if not exists {STATE} (market_id bigint, snapset text, n int,"
                    f" done_at timestamptz default now(), primary key (market_id, snapset))")
        cur.execute("select to_regclass(%s)", (TABLE,))
        exists = cur.fetchone()[0] is not None
    conn.commit()
    if not exists:
        cols = _columns(conn, where)
        with conn.cursor() as cur:
            cur.execute(f"create table {TABLE} (" + ", ".join(f"{a} {b}" for a, b in cols) + ")")
        conn.commit()
    with conn.cursor() as cur:
        cur.execute("select column_name from information_schema.columns where table_name = %s"
                    " order by ordinal_position", (TABLE,))
        return [r[0] for r in cur.fetchall()]


def _copy(conn, df: pd.DataFrame, cols: list[str]):
    df = df.rename(columns=str.lower)
    missing = set(cols) - set(df.columns)
    assert not missing, f"columns missing from computed frame: {sorted(missing)}"
    df = df.reindex(columns=cols)
    df["cutoff_ts"] = pd.to_datetime(df["cutoff_ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00")
    for c in cols:
        if c in BOOL:
            df[c] = df[c].map({True: "t", False: "f"})
    num = [c for c in cols if c not in TEXT | BOOL | {"cutoff_ts", "snapshot_id", "market_id"}]
    df[num] = df[num].astype(float).replace([np.inf, -np.inf], np.nan)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N", float_format="%.7g")
    buf.seek(0)
    with conn.cursor() as cur:
        with cur.copy(f"copy {TABLE} ({', '.join(cols)}) from stdin with (format csv, null '\\N')") as cp:
            cp.write(buf.getvalue())


_G = {}


def _init(where, snapset, cols):
    _G.update(conn=connect(), where=where, snapset=snapset, cols=cols)


def _work(mids: list[int]) -> int:
    conn, where = _G["conn"], _G["where"]
    req = pd.read_sql_query(
        f"select snapshot_id, market_id, cutoff_ts from eval_snapshots where market_id = any(%(m)s)"
        f" and {where} and snapshot_id not in (select snapshot_id from {TABLE} where market_id = any(%(m)s))",
        conn, params={"m": mids})
    if len(req):
        w = _wide(conn, req)
        _copy(conn, w, _G["cols"])
    with conn.cursor() as cur:
        for m in mids:
            cur.execute(f"insert into {STATE} (market_id, snapset, n) values (%s, %s, %s)"
                        f" on conflict do nothing", (m, _G["snapset"], int((req.market_id == m).sum())))
    conn.commit()
    return len(req)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="grid24", choices=list(SETS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only this many pending markets (testing)")
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    where = SETS[a.set]
    conn = connect()
    cols = _ensure(conn, where, a.rebuild)
    pend = pd.read_sql_query(
        f"select market_id, count(*) n from eval_snapshots where {where} and market_id not in"
        f" (select market_id from {STATE} where snapset = %(s)s) group by 1 order by 2 desc",
        conn, params={"s": a.set})
    if a.limit:
        pend = pend.head(a.limit)
    # chunks of ~400 snapshots (big markets alone)
    chunks, cur, acc = [], [], 0
    for m, n in pend.itertuples(index=False):
        cur.append(int(m)); acc += n
        if acc >= 400:
            chunks.append(cur); cur, acc = [], 0
    if cur:
        chunks.append(cur)
    total = int(pend.n.sum())
    print(f"{len(pend)} markets / {total} snapshots pending in {len(chunks)} chunks", flush=True)
    t0, done = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(where, a.set, cols)) as pool:
        for i, k in enumerate(pool.imap_unordered(_work, chunks)):
            done += k
            if i % 20 == 0 or i == len(chunks) - 1:
                el = time.time() - t0
                print(f"chunk {i+1}/{len(chunks)} snapshots {done}/{total} {el:.0f}s"
                      f" eta {el / max(done, 1) * (total - done):.0f}s", flush=True)
    with conn.cursor() as cur:
        cur.execute(f"analyze {TABLE}")
        cur.execute(f"select count(*), pg_size_pretty(pg_total_relation_size('{TABLE}')) from {TABLE}")
        print("done", cur.fetchone())
    conn.commit()


if __name__ == "__main__":
    main()

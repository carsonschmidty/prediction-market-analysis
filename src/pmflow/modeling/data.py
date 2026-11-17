"""Modeling frame: eval_snapshots (primary set, 24h grid) joined with feat_snapshot_features and the
outcome-free columns of feat_weak_labels, plus engineered features and walk-forward folds.

Rules enforced here:
* rows: ``is_live AND split IN ('train','val','test')`` that have a feat_snapshot_features row (24h grid);
* ``hours_to_close_diag`` / ``hours_to_end_date_diag`` are never loaded; ``hours_to_end`` (final Gamma
  end_date, may have been edited after the fact) is kept only for one diagnostic ablation, never in a candidate;
* every ``lbl_*`` column (uses the outcome) is loaded into a separate frame (`load_aux_labels`) and never into
  the feature frame;
* ``assert_fit_rows`` must be called on every frame a model is fitted on; it raises on any test/purged row.

Build: ``python -m pmflow.modeling.data`` (writes data/modeling/snap.parquet, gitignored, ~60 MB).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = os.environ.get("PMFLOW_DSN", "postgresql://pmflow:pmflow@localhost:5432/pmflow")
REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "data" / "modeling" / "snap.parquet"
ART = REPO / "artifacts" / "modeling"
SEED = 20261117

FIT_SPLITS = ("train",)          # splits a model may be fitted on during development
VAL_SPLIT_START = pd.Timestamp("2026-04-01", tz="UTC")
TEST_SPLIT_START = pd.Timestamp("2026-07-01", tz="UTC")

# walk-forward folds inside train (by event_close). Fold k: fit on train events with event_close < lo_k,
# score events with lo_k <= event_close < hi_k and cutoff >= lo_k (mirrors how val is constructed).
WF_BOUNDS = ["2025-01-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-02-15", "2026-04-01"]

SNAP_SQL = """
SELECT s.snapshot_id, s.market_id, s.event_id, s.universe, s.geo_subtype AS geo_subtype_eval, s.neg_risk, s.y,
       s.cutoff_ts, s.price_yes, s.vwap_6h, s.vwap_24h, s.last_trade_age_h, s.n_trades_7d, s.usd_7d,
       s.n_trades_cum, s.usd_cum, s.pre_trade_window_market, s.split, e.event_close
FROM eval_snapshots s JOIN eval_event_splits e USING (event_id)
WHERE s.is_live AND s.split IN ('train','val','test')
"""
FORBIDDEN = ("hours_to_close_diag", "hours_to_end_date_diag")


def connect():
    return psycopg.connect(DSN, autocommit=True)


def _q(sql: str) -> pd.DataFrame:
    with connect() as c:
        cur = c.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def build_cache(path: Path = CACHE) -> pd.DataFrame:
    snaps = _q(SNAP_SQL)
    feat = _q("SELECT * FROM feat_snapshot_features")
    feat = feat.drop(columns=[c for c in ("market_id", "cutoff_ts") if c in feat])
    wl_cols = _q("SELECT column_name FROM information_schema.columns WHERE table_name='feat_weak_labels'").column_name
    keep = ["snapshot_id"] + [c for c in wl_cols if c.startswith(("pat_", "cond_"))]
    wl = _q(f"SELECT {', '.join(keep)} FROM feat_weak_labels")
    meta = _q("SELECT market_id, question FROM feat_market_meta")
    df = snaps.merge(feat, on="snapshot_id", how="inner").merge(wl, on="snapshot_id", how="left")
    df = df.merge(meta, on="market_id", how="left")
    for c in df.columns:
        if df[c].dtype == object and c not in ("universe", "geo_subtype_eval", "split", "category", "geo_subtype",
                                               "question"):
            try:
                df[c] = pd.to_numeric(df[c])
            except (TypeError, ValueError):
                pass
    for c in df.columns:
        if str(df[c].dtype).startswith("float"):
            df[c] = df[c].astype("float32")
    assert not any(c in df.columns for c in FORBIDDEN)
    assert not any(c.startswith("lbl_") for c in df.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_aux_labels() -> pd.DataFrame:
    """Outcome-based weak labels (LABEL ONLY; never features). Used only as an auxiliary target on TRAIN."""
    return _q("SELECT snapshot_id, " + ", ".join(
        f"lbl_l2_{L}h, lbl_n2_{L}h, pat_possible_{L}h" for L in (1, 6, 24, 72)) + " FROM feat_weak_labels")


def load_frame(path: Path = CACHE) -> pd.DataFrame:
    if not path.exists():
        build_cache(path)
    df = pd.read_parquet(path)
    df["cutoff_ts"] = pd.to_datetime(df["cutoff_ts"], utc=True)
    df["event_close"] = pd.to_datetime(df["event_close"], utc=True)
    df = df.sort_values("snapshot_id").reset_index(drop=True)
    return add_engineered(df)


# ------------------------------------------------------------------ guards


class ForbiddenFitRows(AssertionError):
    pass


def assert_fit_rows(df: pd.DataFrame, allowed=FIT_SPLITS) -> None:
    """Raise if a frame about to be used for fitting contains anything but the allowed splits, or any row
    with cutoff in the test period. Called by every fit routine in this package."""
    bad = ~df["split"].isin(list(allowed))
    if bad.any():
        raise ForbiddenFitRows(f"{int(bad.sum())} rows from splits {sorted(df.loc[bad, 'split'].unique())} in fit")
    if "cutoff_ts" in df and (df["cutoff_ts"] >= TEST_SPLIT_START).any():
        raise ForbiddenFitRows("rows with cutoff in the test period in fit")


def no_test(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that may be scored during development (train/val only)."""
    return df[df["split"].isin(["train", "val"])]


# ------------------------------------------------------------------ features

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def slog(x):
    """Signed log1p for heavy-tailed signed USD quantities."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    new = {}
    new["price_logit"] = logit(d.price_yes)
    for L in (1, 6, 24, 72):
        dom = d[f"dom_{L}h"].fillna(0).to_numpy(float)
        new[f"s_nsn_{L}h"] = slog(d[f"nsn_{L}h"].fillna(0))                       # signed net flow toward YES
        new[f"s_os_{L}h"] = dom * d[f"os_{L}h"].fillna(0)                          # signed one-sidedness
        new[f"s_fresh_nsn_{L}h"] = slog(d[f"fresh_nsn_{L}h"].fillna(0))           # fresh-wallet signed flow
        new[f"s_fws72_{L}h"] = dom * d[f"fws72_{L}h"].fillna(0)
        new[f"log_gv_{L}h"] = np.log1p(d[f"gv_{L}h"].fillna(0))
        for p in ("strong", "possible", "loose"):
            new[f"s_pat_{p}_{L}h"] = dom * d[f"pat_{p}_{L}h"].fillna(False).astype(float)
    for c in ("dom_newmkt_share", "top3_share", "conc_top3", "cluster_share", "whale_hist_share",
              "hist_good_share", "fresh_conc_share", "dom_winrate", "share_ge1k", "urg_share_1c"):
        new[f"s_{c}_24h"] = d["dom_24h"].fillna(0).to_numpy(float) * d[f"{c}_24h"].to_numpy(float)
    new["s_pat_loose_any"] = np.sign(sum(new[f"s_pat_loose_{L}h"] for L in (1, 6, 24, 72)))
    new["s_pat_possible_any"] = np.sign(sum(new[f"s_pat_possible_{L}h"] for L in (1, 6, 24, 72)))
    new["log_hours_since_open"] = np.log1p(d.hours_since_open.clip(lower=0))
    new["log_hours_since_first_trade"] = np.log1p(d.hours_since_first_trade.clip(lower=0))
    new["log_n_wallets_total"] = np.log1p(d.n_wallets_total)
    new["log_n_trades_total"] = np.log1p(d.n_trades_total)
    new["log_last_trade_age_h"] = np.log1p(d.last_trade_age_h.clip(lower=0))
    new["log_max_trade_24h"] = np.log1p(d.max_trade_24h.fillna(0))
    new["log_n_new_wallets_dom_24h"] = np.log1p(d.n_new_wallets_dom_24h.fillna(0))
    new["neg_risk_f"] = d.neg_risk.astype(float)
    new["in_scope_f"] = (d.universe == "in_scope").astype(float)
    new["geo_code"] = pd.Categorical(d.geo_subtype.fillna("none")).codes.astype(float)
    new["cat_code"] = pd.Categorical(d.category.fillna("none")).codes.astype(float)
    new["vwap24_minus_px"] = (d.vwap_24h.astype(float) - d.price_yes.astype(float)).fillna(0)
    out = pd.concat([d, pd.DataFrame(new, index=d.index).astype("float32")], axis=1)
    return out


# Pre-specified feature groups (fixed before any val result was looked at). Price enters every model as the
# fixed offset; GBM variants additionally see price_logit so they can learn price x flow interactions.
FLOW = ([f"s_nsn_{L}h" for L in (6, 24, 72)] + [f"s_os_{L}h" for L in (6, 24, 72)] +
        [f"log_gv_{L}h" for L in (6, 24, 72)] +
        ["gv_z_24h", "nsn_z_24h", "wl_z_nsn_24h", "gv_ratio_base_24h", "dp_24h", "dp_6h", "dp_72h",
         "dp_dom_24h", "urg_prem_24h", "impact_per_10k_24h", "vwap24_minus_px", "s_share_ge1k_24h",
         "top5_trade_share_24h", "log_max_trade_24h", "s_urg_share_1c_24h"])
WALLET = ([f"s_fresh_nsn_{L}h" for L in (6, 24, 72)] + [f"s_fws72_{L}h" for L in (6, 24, 72)] +
          ["fresh_nsn_share_24h", "dom_age_med_d_24h", "s_dom_newmkt_share_24h", "dom_log_prior_mkts_24h",
           "s_dom_winrate_24h", "dom_winrate_cov_24h", "s_top3_share_24h", "s_conc_top3_24h",
           "s_cluster_share_24h", "s_whale_hist_share_24h", "s_hist_good_share_24h", "s_fresh_conc_share_24h",
           "log_n_new_wallets_dom_24h", "max_cluster_24h", "wallet_age_censored"])
FLAGS = ([f"s_pat_{p}_{L}h" for p in ("strong", "possible", "loose") for L in (1, 6, 24, 72)] +
         ["s_pat_loose_any", "s_pat_possible_any"])
CONTEXT = ["log_hours_since_open", "log_hours_since_first_trade", "log_gv_total", "log_gv_7d",
           "log_n_wallets_total", "log_n_trades_total", "log_last_trade_age_h", "burst_b_24h",
           "peak10m_share_24h", "neg_risk_f", "in_scope_f", "geo_code", "cat_code", "market_history_truncated",
           "dow_utc"]
DIAG_ONLY = ["hours_to_end"]      # never in a candidate (end_date is the final Gamma value)

# the simplest model: ridge logistic regression on the price-logit offset + a handful of features
RIDGE_FEATURES = ["s_nsn_24h", "s_fresh_nsn_24h", "s_os_24h", "gv_z_24h", "s_pat_loose_any",
                  "s_pat_possible_any"]

GROUPS = {
    "price_only": [],
    "flow": FLOW,
    "wallet": WALLET,
    "flags": FLAGS,
    "context": CONTEXT,
    "all": FLOW + WALLET + FLAGS + CONTEXT,
}


# ------------------------------------------------------------------ folds


def walk_forward_folds(df: pd.DataFrame):
    """Yield (k, fit_mask, score_mask) over TRAIN rows only. Event-grouped by construction (an event's
    membership is decided by its event_close)."""
    tr = df["split"].eq("train").to_numpy()
    ec = df["event_close"]
    for k in range(len(WF_BOUNDS) - 1):
        lo, hi = pd.Timestamp(WF_BOUNDS[k], tz="UTC"), pd.Timestamp(WF_BOUNDS[k + 1], tz="UTC")
        fit = tr & (ec < lo).to_numpy()
        score = tr & ((ec >= lo) & (ec < hi) & (df["cutoff_ts"] >= lo)).to_numpy()
        yield k, fit, score


def weights(df: pd.DataFrame, scheme: str = "market") -> np.ndarray:
    """Training weights: 'market' (each market sums to 1), 'event' (each event sums to 1), 'snapshot'.
    Rescaled to mean 1."""
    if scheme == "snapshot":
        w = np.ones(len(df))
    else:
        n_m = df.groupby("market_id")["market_id"].transform("size").to_numpy(float)
        w = 1.0 / n_m
        if scheme == "event":
            k = df.groupby("event_id")["market_id"].transform("nunique").to_numpy(float)
            w = w / k
    return w / w.mean()


if __name__ == "__main__":
    d = build_cache()
    print(d.shape, d.split.value_counts().to_dict())

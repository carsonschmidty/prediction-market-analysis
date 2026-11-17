"""Tests for the market-anchoring (offset) logic and for the no-test-rows-in-fitting guards.

Run: .venv/bin/python -m pytest src/pmflow/modeling/tests -q
Toy arrays below are unit-test fixtures only (never used as data for any result).
"""
import numpy as np
import pandas as pd
import pytest

from pmflow.modeling import data as D
from pmflow.modeling.evaluate import DevScoringRefused, score
from pmflow.modeling.models import AnchoredLGBM, OffsetLogit


def _toy(n=4000, split="train", seed=0):
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < price).astype(int)
    df = pd.DataFrame(dict(x1=rng.normal(size=n), x2=rng.normal(size=n), price_yes=price, y=y,
                           split=split, cutoff_ts=pd.Timestamp("2025-06-01", tz="UTC"),
                           event_id=rng.integers(0, 200, n), market_id=rng.integers(0, 400, n),
                           universe="in_scope"))
    return df


def test_offset_logit_strong_penalty_returns_price():
    df = _toy()
    m = OffsetLogit(l2=1e6).fit(df[["x1", "x2"]].to_numpy(), D.logit(df.price_yes), df.y, frame=df)
    p = m.predict_proba(df[["x1", "x2"]].to_numpy(), D.logit(df.price_yes))
    assert np.max(np.abs(p - df.price_yes)) < 1e-3


def test_offset_logit_intercept_only_matches_closed_form_direction():
    df = _toy()
    y = df.y.to_numpy().copy()
    y[: len(y) // 4] = 1             # toy fixture: push labels toward YES
    m = OffsetLogit(l2=0.0, penalize_intercept=False).fit(None, D.logit(df.price_yes), y)
    assert m.intercept_ > 0.1
    # first-order condition: sum(p - y) == 0 at the optimum
    p = m.predict_proba(None, D.logit(df.price_yes))
    assert abs(np.mean(p - y)) < 1e-4


def test_offset_logit_free_slope_recovers_one_when_price_is_right():
    df = _toy(n=40000)
    m = OffsetLogit(l2=0.0, free_slope=True, penalize_intercept=False).fit(None, D.logit(df.price_yes), df.y)
    assert abs(m.slope_ - 1) < 0.08 and abs(m.intercept_) < 0.08


def test_lgbm_zero_rounds_is_price():
    df = _toy()
    m = AnchoredLGBM(["x1", "x2"], n_rounds=0).fit(df, D.logit(df.price_yes), df.y, np.ones(len(df)))
    np.testing.assert_allclose(m.predict_proba(df, D.logit(df.price_yes)), df.price_yes, atol=1e-6)


def test_lgbm_init_score_is_offset_not_in_raw_score():
    """LightGBM raw_score excludes init_score; the model must add logit(price) back. With noise features
    and labels drawn from price, the learned residual must be ~0 and p ~ price."""
    df = _toy(n=20000)
    off = D.logit(df.price_yes)
    m = AnchoredLGBM(["x1", "x2"], n_rounds=50).fit(df, off, df.y, np.ones(len(df)))
    r = m.resid_logit(df)
    assert np.abs(r).mean() < 0.1                      # residual near zero, not ~ mean(logit price)
    p = m.predict_proba(df, off)
    assert np.abs(p - df.price_yes).mean() < 0.02


def test_lgbm_learns_residual_signal_on_top_of_offset():
    df = _toy(n=20000)
    off = D.logit(df.price_yes)
    z = off + 1.0 * df.x1.to_numpy()
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=len(df)) < D.expit(z)).astype(int)
    m = AnchoredLGBM(["x1", "x2"], n_rounds=300).fit(df, off, y, np.ones(len(df)))
    r = m.resid_logit(df)
    assert np.corrcoef(r, df.x1)[0, 1] > 0.9


def test_fit_guards_refuse_test_rows():
    for sp in ("test", "val", "purged_test", "purged_val"):
        df = _toy(n=500, split=sp)
        with pytest.raises(D.ForbiddenFitRows):
            D.assert_fit_rows(df)
        with pytest.raises(D.ForbiddenFitRows):
            AnchoredLGBM(["x1"], n_rounds=5).fit(df, D.logit(df.price_yes), df.y, np.ones(len(df)))
        with pytest.raises(D.ForbiddenFitRows):
            OffsetLogit().fit(df[["x1"]].to_numpy(), D.logit(df.price_yes), df.y, frame=df)
    df = _toy(n=500)
    df.loc[0, "cutoff_ts"] = pd.Timestamp("2026-07-02", tz="UTC")
    with pytest.raises(D.ForbiddenFitRows):
        D.assert_fit_rows(df)


def test_dev_scoring_refuses_test_rows():
    df = _toy(n=500, split="test")
    with pytest.raises(DevScoringRefused):
        score(df, df.price_yes, "x")


@pytest.fixture(scope="module")
def frame():
    try:
        return D.load_frame()
    except Exception as e:  # pragma: no cover - needs the DB/cache
        pytest.skip(f"modeling frame unavailable: {e}")


def test_walk_forward_folds_are_train_only_and_time_ordered(frame):
    for k, fit, sc in D.walk_forward_folds(frame):
        f, s = frame[fit], frame[sc]
        assert set(f.split) == {"train"} and set(s.split) == {"train"}
        assert f.event_close.max() < s.cutoff_ts.min()           # every fit label known before any scored cutoff
        assert not set(f.event_id) & set(s.event_id)


def test_frame_has_no_outcome_or_diag_columns(frame):
    assert not [c for c in frame.columns if c.startswith("lbl_")]
    assert not [c for c in frame.columns if c in D.FORBIDDEN]
    for g, cols in D.GROUPS.items():
        assert "hours_to_end" not in cols and "y" not in cols


def test_val_train_split_times(frame):
    tr = frame[frame.split == "train"]
    va = frame[frame.split == "val"]
    assert tr.event_close.max() < D.VAL_SPLIT_START <= va.cutoff_ts.min()


def test_frozen_prediction_files_cover_all_splits():
    import glob
    files = glob.glob(str(D.ART / "preds_*.parquet"))
    if not files:
        pytest.skip("no frozen predictions yet")
    ref = D.load_frame()[["snapshot_id", "split"]]
    for f in files:
        p = pd.read_parquet(f)
        assert p.snapshot_id.is_unique and p.p_model.between(0, 1).all() and p.p_model.notna().all()
        assert set(p.snapshot_id) == set(ref.snapshot_id), f
        assert set(p.split) == {"train", "val", "test"}
        assert set(p.flag.unique()) <= {-1, 0, 1}


def test_no_test_rows_reach_any_fit(monkeypatch):
    """Instrument the two fit entry points and run the stage-2 fitting helpers on the real frame: every frame
    that reaches a fit must be TRAIN only."""
    from pmflow.modeling import models as MM
    from pmflow.modeling import stage2_tabular as S2
    try:
        d = D.no_test(D.load_frame()).reset_index(drop=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(str(e))
    seen = []
    orig = D.assert_fit_rows

    def spy(df, allowed=D.FIT_SPLITS):
        seen.append(set(df["split"]))
        return orig(df, allowed)
    monkeypatch.setattr(MM, "assert_fit_rows", spy)
    full = D.load_frame()                     # includes val + test rows
    S2.fit_lgbm_full(full, ["price_logit", "s_nsn_24h"], 5)
    S2.fit_ridge_full(full, D.RIDGE_FEATURES, 1e-2)
    S2.ridge_wf(d, D.RIDGE_FEATURES, l2_grid=[1e-2])
    assert seen and all(s == {"train"} for s in seen)

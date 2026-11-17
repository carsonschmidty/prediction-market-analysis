"""Market-anchored model primitives.

Every model here predicts logit p = logit(price) + f(x), where logit(price) is a fixed offset (coefficient 1)
and f is shrunk toward 0, so the default answer is "the market is right".

* `OffsetLogit`: L2-penalised logistic regression with offset (scipy L-BFGS). Intercept is penalised too
  unless `penalize_intercept=False`.
* `AnchoredLGBM`: LightGBM binary with ``init_score = logit(price)``; predictions add the offset back
  (LightGBM's ``predict(raw_score=True)`` returns the tree sum only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize

from .data import SEED, assert_fit_rows, expit

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=4, max_depth=2, min_data_in_leaf=400,
                  min_sum_hessian_in_leaf=5.0, lambda_l2=50.0, feature_fraction=0.6, bagging_fraction=0.7,
                  bagging_freq=1, max_bin=63, num_threads=3, verbose=-1, seed=SEED, deterministic=True,
                  force_col_wise=True)


def _wll(y, z, w):
    """weighted mean log loss for logits z"""
    return float(np.sum(w * (np.logaddexp(0, z) - y * z)) / np.sum(w))


@dataclass
class Standardizer:
    clip: float = 5.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, w=None):
        w = np.ones(len(X)) if w is None else w
        Xm = np.ma.masked_invalid(X)
        self.mu = np.ma.average(Xm, axis=0, weights=np.broadcast_to(w[:, None], X.shape)).filled(0.0)
        var = np.ma.average((Xm - self.mu) ** 2, axis=0, weights=np.broadcast_to(w[:, None], X.shape)).filled(1.0)
        self.sd = np.where(np.sqrt(var) > 1e-9, np.sqrt(var), 1.0)
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.clip(Z, -self.clip, self.clip)


@dataclass
class OffsetLogit:
    l2: float = 1e-3
    fit_intercept: bool = True
    penalize_intercept: bool = True
    free_slope: bool = False          # if True, the offset gets a free coefficient (Platt-type control)
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    slope_: float = 1.0
    scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, X, offset, y, w=None, frame: pd.DataFrame | None = None):
        if frame is not None:
            assert_fit_rows(frame)
        X = np.zeros((len(y), 0)) if X is None else np.asarray(X, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        Z = self.scaler.fit(X, w).transform(X) if X.shape[1] else X
        y = np.asarray(y, float)
        o = np.asarray(offset, float)
        p = Z.shape[1]
        n_extra = int(self.fit_intercept) + int(self.free_slope)

        def unpack(th):
            b = th[:p]
            i = p
            a = th[i] if self.fit_intercept else 0.0
            i += int(self.fit_intercept)
            s = 1.0 + th[i] if self.free_slope else 1.0
            return b, a, s

        def f(th):
            b, a, s = unpack(th)
            z = s * o + a + Z @ b
            pr = expit(z)
            W = w.sum()
            loss = np.sum(w * (np.logaddexp(0, z) - y * z)) / W
            r = w * (pr - y) / W
            g = np.concatenate([Z.T @ r, [r.sum()] if self.fit_intercept else [], [r @ o] if self.free_slope else []])
            pen = self.l2 * np.sum(b ** 2)
            g[:p] += 2 * self.l2 * b
            if self.fit_intercept and self.penalize_intercept:
                pen += self.l2 * a ** 2
                g[p] += 2 * self.l2 * a
            if self.free_slope:
                pen += self.l2 * (s - 1.0) ** 2
                g[-1] += 2 * self.l2 * (s - 1.0)
            return loss + pen, g

        th0 = np.zeros(p + n_extra)
        res = optimize.minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.coef_, self.intercept_, self.slope_ = unpack(res.x)
        return self

    def resid_logit(self, X, offset):
        X = np.zeros((len(offset), 0)) if X is None else np.asarray(X, float)
        Z = self.scaler.transform(X) if X.shape[1] else X
        return (self.slope_ - 1.0) * np.asarray(offset, float) + self.intercept_ + Z @ self.coef_

    def predict_proba(self, X, offset):
        return expit(np.asarray(offset, float) + self.resid_logit(X, offset))


@dataclass
class AnchoredLGBM:
    features: list
    n_rounds: int = 300
    params: dict = field(default_factory=lambda: dict(LGB_PARAMS))
    monotone: dict | None = None
    booster: lgb.Booster | None = None

    def _p(self):
        p = dict(self.params)
        if self.monotone:
            p["monotone_constraints"] = [int(self.monotone.get(f, 0)) for f in self.features]
            p["monotone_constraints_method"] = "advanced"
        return p

    def dataset(self, df, offset, y, w):
        return lgb.Dataset(df[self.features].to_numpy(np.float32), label=np.asarray(y, float), weight=w,
                           init_score=np.asarray(offset, float), feature_name=list(self.features),
                           free_raw_data=False)

    def fit(self, df: pd.DataFrame, offset, y, w, valid=None, record=None):
        assert_fit_rows(df)
        if self.n_rounds == 0 or len(self.features) == 0:
            self.booster = None
            return self
        ds = self.dataset(df, offset, y, w)
        kw = {}
        if valid is not None:
            vds = self.dataset(*valid)
            kw = dict(valid_sets=[vds], valid_names=["v"], callbacks=[lgb.record_evaluation(record)])
        p = self._p()
        p["metric"] = "binary_logloss"
        self.booster = lgb.train(p, ds, num_boost_round=self.n_rounds, **kw)
        return self

    def resid_logit(self, df, n_rounds=None):
        if self.booster is None:
            return np.zeros(len(df))
        return self.booster.predict(df[self.features].to_numpy(np.float32), raw_score=True,
                                    num_iteration=n_rounds or self.n_rounds)

    def predict_proba(self, df, offset, n_rounds=None):
        return expit(np.asarray(offset, float) + self.resid_logit(df, n_rounds))

    def shap(self, df):
        return self.booster.predict(df[self.features].to_numpy(np.float32), pred_contrib=True)

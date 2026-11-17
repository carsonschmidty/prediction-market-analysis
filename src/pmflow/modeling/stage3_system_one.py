"""Stage 3 (EXPLORATORY, gate NOT passed): small "system one" text encoder on serialised snapshots.

    python -m pmflow.modeling.stage3_system_one embed      # serialise + embed all rows (~25-30 min CPU)
    python -m pmflow.modeling.stage3_system_one heads      # anchored heads on frozen embeddings (OOF + val)
    python -m pmflow.modeling.stage3_system_one finetune   # time-boxed fine-tune on a train subsample -> val
    python -m pmflow.modeling.stage3_system_one all

Backbone: the ModernBERT (Ettin-32m) encoder inside knowledgator/gliclass-instruct-edge-v1.0 (Apache-2.0,
revision pinned below), run with our JAX implementation (`encoder_jax`, validated against onnxruntime). The
GLiClass label-scoring head is not used; a fresh binary head models the residual over the price logit
(market-anchored, like stage 2). This is the "plain sequence-classification head on the same backbone"
substitute allowed by the brief, because PyTorch could not be installed (CUDA-only wheel, disk).

Input text, built only from pre-cutoff data: market question (question field only; descriptions are NOT used),
category / sub-type, time since open, price bucket, and <=6 binned verbalised flow descriptors. No scheduled end
date (same rule as the tabular candidates), no trade-sequence serialisation (not done: time box).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from pmflow.evaluation import metrics as M

from . import data as D
from .evaluate import score
from .models import OffsetLogit, _wll

HF = D.REPO / "data" / "modeling" / "hf" / "knowledgator__gliclass-instruct-edge-v1.0"
REVISION = "727be8a417f6a7718e591b025e07054c146d8139"
EMB = D.REPO / "data" / "modeling" / "s1_emb.npy"
EMB_IDS = D.REPO / "data" / "modeling" / "s1_emb_ids.npy"
QEMB = D.REPO / "data" / "modeling" / "s1_qemb.parquet"
MAXLEN = 128
N_PC = 32


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ serialisation


def _bucket(x, edges, labels):
    if x is None or not np.isfinite(x):
        return None
    for e, l in zip(edges, labels):
        if x < e:
            return l
    return labels[-1]


def _s(v, default):
    return v if isinstance(v, str) and v else default


def _side(v):
    return "YES" if v > 0 else "NO"


def describe(r) -> str:
    """Binned, verbalised descriptors of one snapshot (all inputs are strictly pre-cutoff features)."""
    px = float(r.price_yes)
    parts = [f"Question: {str(r.question).strip()}",
             f"Category: {_s(r.category, 'unknown')}, {_s(r.geo_subtype, 'general').replace('_', ' ')}."]
    days = float(r.hours_since_open) / 24 if np.isfinite(r.hours_since_open) else np.nan
    parts.append("Market opened " + (_bucket(days, [1, 7, 30, 90], ["less than a day ago", "within the last week",
                                                                 "within the last month", "one to three months ago",
                                                                 "more than three months ago"]) or "at an unknown time") + ".")
    parts.append("The market prices YES as " + _bucket(px, [0.1, 0.3, 0.5, 0.7, 0.9],
                 ["a long shot (under 10%)", "unlikely (10-30%)", "somewhat unlikely (30-50%)",
                  "somewhat likely (50-70%)", "likely (70-90%)", "very likely (over 90%)"]) + ".")
    n24 = r.n_trades_24h if np.isfinite(r.n_trades_24h) else 0
    if n24 == 0:
        parts.append("No trades in the last 24 hours.")
    else:
        sos = float(r.s_os_24h)
        flow = ("strongly toward " + _side(sos)) if abs(sos) >= 0.5 else (
            ("mildly toward " + _side(sos)) if abs(sos) >= 0.2 else "balanced")
        parts.append(f"In the last 24 hours net taker flow was {flow}.")
        z = r.gv_z_24h
        vol = _bucket(z, [-1, 1, 3], ["below usual", "typical", "above usual", "far above usual"])
        if vol:
            parts.append(f"Trading volume was {vol}.")
        fn = r.fresh_nsn_24h
        if np.isfinite(fn) and abs(fn) >= 1000:
            amt = _bucket(abs(fn), [10_000, 100_000], ["under $10k", "$10k-$100k", "over $100k"])
            parts.append(f"Wallets new to these markets were net buyers of {_side(fn)} ({amt}).")
        else:
            parts.append("Wallets new to these markets were not notably active.")
        dp = r.dp_24h
        if np.isfinite(dp):
            mv = _bucket(abs(dp) * 100, [1, 5, 15], ["was flat", "moved a little", "moved substantially",
                                                    "moved sharply"])
            parts.append(f"The YES price {mv}" + ("" if abs(dp) < 0.01 else (" up" if dp > 0 else " down")) + ".")
        mt = r.max_trade_24h
        if np.isfinite(mt):
            parts.append("Largest single trade " + _bucket(mt, [1_000, 10_000, 100_000],
                         ["under $1k", "$1k-$10k", "$10k-$100k", "over $100k"]) + ".")
    pat = r.s_pat_loose_any
    if np.isfinite(pat) and pat != 0:
        parts.append(f"An unusual one-sided burst of buying toward {_side(pat)} was detected.")
    return " ".join(parts)


# ------------------------------------------------------------------ embedding


def _tokenize(texts, tok):
    enc = tok.encode_batch(texts)
    return [e.ids[:MAXLEN] for e in enc]


def _embed_ids(params, id_lists, bs=64):
    import jax.numpy as jnp
    from .encoder_jax import embed_jit
    out = np.zeros((len(id_lists), 384), np.float32)
    lens = np.array([len(x) for x in id_lists])
    order = np.argsort(lens)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        T = int(lens[idx].max())
        T = min(MAXLEN, int(np.ceil(T / 16) * 16))
        ids = np.full((bs, T), 50283, np.int32)
        mask = np.zeros((bs, T), np.int32)
        for j, i in enumerate(idx):
            e = id_lists[i][:T]
            ids[j, :len(e)] = e
            mask[j, :len(e)] = 1
        emb = np.asarray(embed_jit(params, jnp.asarray(ids), jnp.asarray(mask)))
        out[idx] = emb[:len(idx), :384]                      # mean-pooled part
        if (s // bs) % 200 == 0:
            log(f"embedded {s}/{len(order)} (T={T})")
    return out


def run_embed():
    from tokenizers import Tokenizer
    from .encoder_jax import load_gliclass_encoder
    d = D.load_frame()
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    params = load_gliclass_encoder(str(HF / "model.safetensors"))
    texts = [describe(r) for r in d.itertuples()]
    lens = [len(x) for x in _tokenize(texts[:2000], tok)]
    log(f"{len(texts)} texts; token length p50={np.median(lens):.0f} p95={np.percentile(lens, 95):.0f} "
        f"max={max(lens)}; example:\n{texts[0]}")
    pd.DataFrame(dict(snapshot_id=d.snapshot_id[:20], text=texts[:20])).to_csv(D.ART / "stage3_text_examples.csv",
                                                                               index=False)
    # question-only (one per market): the contamination / text-only check
    qs = d.drop_duplicates("market_id")[["market_id", "question"]]
    qe = _embed_ids(params, _tokenize([f"Question: {q}" for q in qs.question.astype(str)], tok))
    pd.DataFrame(qe.astype(np.float16), columns=[f"q{i}" for i in range(384)]).assign(
        market_id=qs.market_id.to_numpy()).to_parquet(QEMB, index=False)
    log("question embeddings done")
    E = _embed_ids(params, _tokenize(texts, tok))
    np.save(EMB, E.astype(np.float16))
    np.save(EMB_IDS, d.snapshot_id.to_numpy())
    log("snapshot embeddings done")


# ------------------------------------------------------------------ heads on frozen embeddings


def _pca_fit(X, k=N_PC, seed=D.SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(30000, len(X)), replace=False)
    mu = X[idx].mean(0)
    _, _, Vt = np.linalg.svd(X[idx] - mu, full_matrices=False)
    return mu, Vt[:k]


def load_embedded_frame():
    d = D.load_frame()
    E = np.load(EMB).astype(np.float32)
    ids = np.load(EMB_IDS)
    assert (ids == d.snapshot_id.to_numpy()).all()
    qe = pd.read_parquet(QEMB)
    Q = d[["market_id"]].merge(qe, on="market_id", how="left").drop(columns="market_id").to_numpy(np.float32)
    return d, E, Q


def run_heads(B=1000):
    from .stage2_tabular import fit_ridge_full, ridge_wf
    d, E, Q = load_embedded_frame()
    dev_m = d.split.isin(["train", "val"]).to_numpy()
    tr_m = (d.split == "train").to_numpy()
    res, sel = [], {}
    resid_cols = {}
    for name, X in (("s1_text_only", Q), ("s1_text_desc", E)):
        mu, V = _pca_fit(X[tr_m])                          # PCA fitted on TRAIN rows only
        P = (X - mu) @ V.T
        cols = [f"pc{i}" for i in range(P.shape[1])]
        dd = pd.concat([d, pd.DataFrame(P, columns=cols, index=d.index)], axis=1)
        dev = dd[dev_m].reset_index(drop=True)
        for fi in (True, False):
            nm = name + ("" if fi else "_noint")
            l2, oof, t = ridge_wf(dev, cols, l2_grid=[1.0, 1e-1, 1e-2, 1e-3], fit_intercept=fi)
            m = fit_ridge_full(dev, cols, l2, fit_intercept=fi)
            r_all = m.resid_logit(dd[cols].to_numpy(float), dd.price_logit)      # all splits (for freezing)
            va = dev[dev.split == "val"]
            r_val = m.resid_logit(va[cols].to_numpy(float), va.price_logit)
            sel[nm] = dict(l2=l2, intercept=float(m.intercept_), wf=t.to_dict("records"))
            res.append(score(va, D.expit(va.price_logit + r_val), nm, B=B).assign(where="val"))
            ok = np.isfinite(oof)
            so = dev[ok]
            res.append(score(so, D.expit(so.price_logit + oof[ok]), nm, B=B, weighting=("market", "snapshot"))
                       .assign(where="train_oof"))
            resid_cols[nm] = (oof, r_all)
            log(f"{nm}: l2={l2}")
    # auxiliary informed-flow head (weak): P(lbl_L2 at any lookback | text) fitted on TRAIN only, no offset
    aux = D.load_aux_labels()
    lab = d[["snapshot_id"]].merge(aux, on="snapshot_id", how="left")
    yl2 = lab[[f"lbl_l2_{L}h" for L in (1, 6, 24, 72)]].fillna(False).astype(bool).any(axis=1).to_numpy()
    mu, V = _pca_fit(E[tr_m])
    P = (E - mu) @ V.T
    trd = d[tr_m]
    base = np.full(tr_m.sum(), np.log(yl2[tr_m].mean() / (1 - yl2[tr_m].mean())))
    ma = OffsetLogit(l2=1e-2, penalize_intercept=False).fit(P[tr_m], base, yl2[tr_m].astype(float), frame=trd)
    aux_score = ma.predict_proba(P, np.full(len(P), base[0]))
    from sklearn.metrics import roc_auc_score
    va_m = (d.split == "val").to_numpy()
    aux_info = dict(n_pos_train=int(yl2[tr_m].sum()), n_pos_val=int(yl2[va_m].sum()),
                    auc_train_insample=float(roc_auc_score(yl2[tr_m], aux_score[tr_m])),
                    auc_val=float(roc_auc_score(yl2[va_m], aux_score[va_m])) if yl2[va_m].any() else None)
    sel["aux_informed_head"] = aux_info
    log(f"aux head: {aux_info}")
    R = pd.concat(res, ignore_index=True)
    R.to_csv(D.ART / "stage3_heads_results.csv", index=False)
    (D.ART / "stage3_selected.json").write_text(json.dumps(sel, indent=1, default=float))
    out = pd.DataFrame({"snapshot_id": d.snapshot_id})
    dev_ids = d.snapshot_id[dev_m].to_numpy()
    for nm, (oof, r_all) in resid_cols.items():
        out[nm + "_all"] = r_all
        tmp = pd.Series(np.nan, index=d.snapshot_id)
        tmp.loc[dev_ids] = oof
        out[nm + "_oof"] = tmp.to_numpy()
    out["s1_aux_informed_score"] = aux_score
    out.to_parquet(D.REPO / "data" / "modeling" / "stage3_resid.parquet", index=False)
    v = R[(R.where == "val") & (R.weighting == "market")]
    print(v[["model", "universe", "d_brier", "d_brier_lo", "d_brier_hi", "d_ll", "d_ll_lo", "d_ll_hi"]].round(4))


# ------------------------------------------------------------------ time-boxed fine-tune


def run_finetune(n_train=24000, epochs=1, bs=32, lr_enc=2e-5, lr_head=1e-3, time_budget_s=40 * 60, B=1000):
    """Fine-tune the whole encoder + a linear residual head with the price logit as fixed offset, on a
    market-balanced TRAIN subsample; evaluate on VAL. Single seed, single fit (no walk-forward): exploratory."""
    import jax
    import jax.numpy as jnp
    from tokenizers import Tokenizer
    from .encoder_jax import encode, load_gliclass_encoder, mean_pool
    d = D.load_frame()
    tr = d[d.split == "train"]
    D.assert_fit_rows(tr)
    rng = np.random.default_rng(D.SEED)
    # market-balanced subsample: probability proportional to market weight
    w = D.weights(tr, "market")
    idx = rng.choice(len(tr), size=min(n_train, len(tr)), replace=False, p=w / w.sum())
    sub = tr.iloc[idx]
    va = d[d.split == "val"]
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    enc_params = load_gliclass_encoder(str(HF / "model.safetensors"))
    params = dict(enc=enc_params, w=jnp.zeros(384), b=jnp.zeros(()))

    def batches(frame, shuffle):
        ids_l = _tokenize([describe(r) for r in frame.itertuples()], tok)
        order = rng.permutation(len(frame)) if shuffle else np.arange(len(frame))
        for s in range(0, len(order), bs):
            ii = order[s:s + bs]
            T = min(MAXLEN, int(np.ceil(max(len(ids_l[i]) for i in ii) / 16) * 16))
            ids = np.full((bs, T), 50283, np.int32)
            mask = np.zeros((bs, T), np.int32)
            for j, i in enumerate(ii):
                e = ids_l[i][:T]
                ids[j, :len(e)] = e
                mask[j, :len(e)] = 1
            valid = np.zeros(bs, np.float32)
            valid[:len(ii)] = 1
            yield ii, ids, mask, valid

    def resid(p, ids, mask):
        h = mean_pool(encode(p["enc"], ids, mask), mask)
        return h @ p["w"] + p["b"]

    def loss(p, ids, mask, off, y, valid):
        z = off + resid(p, ids, mask)
        return jnp.sum(valid * (jnp.logaddexp(0, z) - y * z)) / jnp.maximum(valid.sum(), 1) + 1e-3 * (p["w"] @ p["w"])

    grad_fn = jax.jit(jax.value_and_grad(loss))
    resid_jit = jax.jit(resid)
    m_ = jax.tree_util.tree_map(jnp.zeros_like, params)
    v_ = jax.tree_util.tree_map(jnp.zeros_like, params)
    lrs = dict(enc=lr_enc, w=lr_head, b=lr_head)
    step, t0, losses = 0, time.time(), []
    y_all, off_all = sub.y.to_numpy(np.float32), sub.price_logit.to_numpy(np.float32)
    stopped = False
    for ep in range(epochs):
        for ii, ids, mask, valid in batches(sub, True):
            off = np.zeros(bs, np.float32); y = np.zeros(bs, np.float32)
            off[:len(ii)], y[:len(ii)] = off_all[ii], y_all[ii]
            L, g = grad_fn(params, ids, mask, off, y, valid)
            step += 1
            b1, b2 = 0.9, 0.999
            m_ = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m_, g)
            v_ = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v_, g)
            new = {}
            for k in params:
                lr = lrs[k]
                new[k] = jax.tree_util.tree_map(
                    lambda p_, mm, vv: p_ - lr * (mm / (1 - b1 ** step)) / (jnp.sqrt(vv / (1 - b2 ** step)) + 1e-8),
                    params[k], m_[k], v_[k])
            params = new
            losses.append(float(L))
            if step % 50 == 0:
                log(f"ft step {step} loss(avg50) {np.mean(losses[-50:]):.4f} "
                    f"price-only loss {_wll(y_all, off_all, np.ones(len(y_all))):.4f} elapsed {time.time() - t0:.0f}s")
            if time.time() - t0 > time_budget_s:
                stopped = True
                break
        if stopped:
            break
    r = np.zeros(len(va))
    for ii, ids, mask, valid in batches(va, False):
        r[ii] = np.asarray(resid_jit(params, ids, mask))[:len(ii)]
    res = score(va, D.expit(va.price_logit + r), "s1_finetune", B=B,
                extra=dict(steps=step, n_seen=step * bs, stopped_by_time=stopped))
    res.to_csv(D.ART / "stage3_finetune_val.csv", index=False)
    pd.DataFrame(dict(step=np.arange(1, len(losses) + 1), loss=losses)).to_csv(D.ART / "stage3_finetune_loss.csv",
                                                                               index=False)
    print(res[["model", "universe", "weighting", "d_brier", "d_brier_lo", "d_brier_hi", "d_ll", "mean_resid_logit"]]
          .round(4))
    log(f"finetune done: {step} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("embed", "all"):
        run_embed()
    if what in ("heads", "all"):
        run_heads()
    if what in ("finetune", "all"):
        run_finetune()

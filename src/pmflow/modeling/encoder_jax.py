"""Minimal ModernBERT encoder in JAX (CPU), so the GLiClass/Ettin backbone can be run and trained without
PyTorch (the only PyTorch wheel installable here is the CUDA build, whose ~4 GB of NVIDIA libraries do not fit
on the shared disk; see docs/modeling.md).

* `load_gliclass_encoder(path)`: the ModernBERT encoder weights inside knowledgator/gliclass-instruct-edge-v1.0
  (``model.encoder_model.*``). The GLiClass label/scorer heads are NOT used; we put our own binary head on top.
* `load_onnx_ettin(path)`: the same architecture's weights from onnx-community/ettin-encoder-32m-ONNX, used only
  by `validate_against_onnx` to check this implementation against onnxruntime (MLM logits must match).

Architecture (HF ``ModernBertModel``): tok embeddings -> LayerNorm(no bias); 10 pre-norm layers (layer 0 has no
attn_norm); attention with RoPE (theta 160k, head_dim 64); every 3rd layer (0,3,6,9) global, others sliding
window |i-j| <= 64; GeGLU MLP ``Wo(gelu(a) * g)`` with ``a, g = Wi(x).chunk(2)``; final LayerNorm.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

CFG = dict(hidden=384, heads=6, layers=10, inter=576, global_every=3, local_window=128, theta=160000.0, eps=1e-5)


def _ln(x, w, eps=CFG["eps"]):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * w


def _rope(T, dim=64, theta=CFG["theta"]):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    f = np.outer(np.arange(T), inv)
    emb = np.concatenate([f, f], -1)
    return jnp.asarray(np.cos(emb), jnp.float32), jnp.asarray(np.sin(emb), jnp.float32)


def _rot_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], -1)


def encode(params, ids, mask, n_layers: int | None = None):
    """ids, mask: (B, T) int32. Returns last hidden state (B, T, H)."""
    B, T = ids.shape
    H, nh = CFG["hidden"], CFG["heads"]
    hd = H // nh
    cos, sin = _rope(T, hd)
    x = _ln(params["emb"][ids], params["emb_norm"])
    pos = jnp.arange(T)
    local = jnp.abs(pos[:, None] - pos[None, :]) <= CFG["local_window"] // 2
    keym = mask[:, None, None, :].astype(bool)
    neg = jnp.finfo(jnp.float32).min
    for i, L in enumerate(params["layers"]):
        h = x if i == 0 else _ln(x, L["attn_norm"])
        qkv = (h @ L["Wqkv"].T).reshape(B, T, 3, nh, hd)
        q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))      # (B, nh, T, hd)
        q = q * cos + _rot_half(q) * sin
        k = k * cos + _rot_half(k) * sin
        s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
        m = keym if i % CFG["global_every"] == 0 else (keym & local[None, None])
        s = jnp.where(m, s, neg)
        a = jax.nn.softmax(s, -1) @ v
        x = x + a.transpose(0, 2, 1, 3).reshape(B, T, H) @ L["Wo"].T
        h = _ln(x, L["mlp_norm"])
        ag = h @ L["Wi"].T
        a_, g_ = ag[..., :CFG["inter"]], ag[..., CFG["inter"]:]
        x = x + (jax.nn.gelu(a_, approximate=False) * g_) @ L["Wmo"].T
    return _ln(x, params["final_norm"])


def mean_pool(hs, mask):
    m = mask[..., None].astype(hs.dtype)
    return (hs * m).sum(1) / jnp.maximum(m.sum(1), 1.0)


@partial(jax.jit, static_argnames=())
def embed_jit(params, ids, mask):
    hs = encode(params, ids, mask)
    return jnp.concatenate([mean_pool(hs, mask), hs[:, 0]], -1)          # [mean ; CLS]


def load_gliclass_encoder(path: str) -> dict:
    from safetensors.numpy import load_file
    w = load_file(path)
    p = "model.encoder_model."
    layers = []
    for i in range(CFG["layers"]):
        q = f"{p}layers.{i}."
        layers.append(dict(attn_norm=w.get(q + "attn_norm.weight", np.ones(CFG["hidden"], np.float32)),
                           Wqkv=w[q + "attn.Wqkv.weight"], Wo=w[q + "attn.Wo.weight"],
                           mlp_norm=w[q + "mlp_norm.weight"], Wi=w[q + "mlp.Wi.weight"], Wmo=w[q + "mlp.Wo.weight"]))
    params = dict(emb=w[p + "embeddings.tok_embeddings.weight"], emb_norm=w[p + "embeddings.norm.weight"],
                  final_norm=w[p + "final_norm.weight"], layers=layers)
    return jax.tree_util.tree_map(jnp.asarray, params)


def load_onnx_ettin(path: str, gate_first: bool = True):
    """Weights of onnx-community/ettin-encoder-32m-ONNX mapped to our layout, plus the MLM head."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path, load_external_data=True)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    lns = []
    for n in m.graph.node:
        if n.op_type == "LayerNormalization":
            sc = n.input[1]
            lns.append((n.name, init.get(sc, consts.get(sc))))
    by = {name: v for name, v in lns}
    get_ln = lambda key: next(v for k, v in lns if key in k)  # noqa: E731
    layers = []
    for i in range(CFG["layers"]):
        q = f"model.layers.{i}."
        Wq, Wk, Wv = (init[q + f"attn.{t}_proj.MatMul.weight"].T for t in "qkv")
        gate, up = init[q + "mlp.gate_proj.MatMul.weight"].T, init[q + "mlp.up_proj.MatMul.weight"].T
        layers.append(dict(attn_norm=init.get(q + "attn_norm_layernorm.weight", np.ones(CFG["hidden"], np.float32)),
                           Wqkv=np.concatenate([Wq, Wk, Wv], 0), Wo=init[q + "attn.o_proj.MatMul.weight"].T,
                           mlp_norm=init[q + "mlp_norm_layernorm.weight"],
                           Wi=np.concatenate([gate, up] if gate_first else [up, gate], 0),
                           Wmo=init[q + "mlp.down_proj.MatMul.weight"].T))
    params = dict(emb=init["model.embed_tokens.weight"], emb_norm=init["model.layers.-1.embeddings.norm_layernorm.weight"],
                  final_norm=get_ln("final_norm") if any("final_norm" in k for k in by) else lns[-2][1],
                  layers=layers)
    head = dict(dense=init["model.lm_head.prediction_head.dense.MatMul.weight"], norm=lns[-1][1],
                bias=init["lm_head.Add.bias"])
    return jax.tree_util.tree_map(jnp.asarray, params), head, [k for k, _ in lns]


def validate_against_onnx(onnx_path: str, tokenizer_path: str, texts=None) -> dict:
    """Run onnxruntime and this implementation on the same inputs; return max |diff| of MLM logits."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    texts = texts or ["Will the US strike Iran before July 1, 2026?",
                      "Market: Will Russia and Ukraine agree a ceasefire by March 31? Price 12 cents. " * 3]
    tok = Tokenizer.from_file(tokenizer_path)
    enc = [tok.encode(t).ids for t in texts]
    T = max(map(len, enc))
    ids = np.full((len(enc), T), 50283, np.int64)
    mask = np.zeros((len(enc), T), np.int64)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = e
        mask[i, :len(e)] = 1
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = sess.run(None, dict(input_ids=ids, attention_mask=mask))[0]
    out = {}
    for gate_first in (True, False):
        params, head, ln_names = load_onnx_ettin(onnx_path, gate_first)
        hs = encode(params, jnp.asarray(ids, jnp.int32), jnp.asarray(mask, jnp.int32))
        h = jax.nn.gelu(hs @ head["dense"], approximate=False)
        h = _ln(h, head["norm"])
        logits = np.asarray(h @ params["emb"].T + head["bias"])
        diff = np.abs(logits - ref)[mask.astype(bool)]
        out[f"gate_first={gate_first}"] = dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                                               ref_scale=float(np.abs(ref).mean()))
    out["T"] = T
    return out

"""Modeling (owner: modeling agent). Market-anchored models of P(resolves YES).

Stages (each has one entry point, fixed seeds):
    python -m pmflow.modeling.data            # build data/modeling/snap.parquet (cached modeling frame)
    python -m pmflow.modeling.stage1_baseline # Baseline A on the 24h-grid modeling rows (train/val only)
    python -m pmflow.modeling.stage2_tabular  # anchored ridge-logit + anchored LightGBM, ablations, placebo
    python -m pmflow.modeling.stage3_system_one   # exploratory frozen/fine-tuned small encoder (see docs)
    python -m pmflow.modeling.stage4_stack    # anchored stack, calibration, flags, frozen predictions

No stage computes a test-split metric. See docs/modeling.md.
"""

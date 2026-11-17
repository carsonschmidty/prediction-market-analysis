import pytest


@pytest.fixture(scope="session")
def conn():
    try:
        from pmflow.features.db import connect
        c = connect(connect_timeout=5)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres unavailable: {e}")
    with c.cursor() as cur:
        cur.execute("select to_regclass('feat_wallet_hist') is not null")
        if not cur.fetchone()[0]:
            pytest.skip("feat_wallet_hist missing: run python -m pmflow.features.base")
    yield c
    c.close()

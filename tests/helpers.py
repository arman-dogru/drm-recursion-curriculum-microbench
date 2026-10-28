from drm.config import Config


def tiny(method="drm", **kw):
    base = dict(task="flood", method=method, seed=1, lr=1e-3, k=2, n_sup=3, phi_budget=24,
                ref_k=2, batch=8, warmup=2, d=16, heads=2, layers=1, n_train=40, n_dev=12,
                n_eval=20, curve_points=2, curve_n=7)
    base.update(kw)
    return Config(**base)

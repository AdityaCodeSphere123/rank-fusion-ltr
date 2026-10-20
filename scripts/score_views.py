"""Score the shipped tree on DEV under each corruption. DEV rows were in the fit,
so the clean number is in-training. The gap from clean to each corruption is the signal.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import lightgbm as lgb

from harness.metrics import mean_ndcg_from_scores
from scripts.data_cache import load_xyq
from scripts.select_qz_views import _corrupt
from scripts.train_model import _expand, _run_codes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    model = lgb.Booster(model_file=os.path.join(ROOT, "submission", "model.txt"))
    width = model.num_feature()
    X, y, q = load_xyq(os.path.join(ROOT, "data", "dev", "vali.txt"))
    q = np.asarray(q)
    codes = _run_codes(q)
    with_z = width == 3 * X.shape[1]
    with_aggs = width == 2 * X.shape[1] + 150
    print(f"width {width} with_z {with_z} with_aggs {with_aggs} rows {len(y)}", flush=True)
    raw = _corrupt(X)
    clean_score = None
    for key, view in raw.items():
        packed = _expand(view, codes, with_aggs, with_z=with_z)
        pred = model.predict(packed, num_threads=4)
        score = mean_ndcg_from_scores(pred, y, q, 10)
        if key == "clean":
            clean_score = score
        gap = 0.0 if clean_score is None else score - clean_score
        print(f"{key:8s} {score:.4f} gap {gap:+.4f}", flush=True)


if __name__ == "__main__":
    main()

"""Generate a small deterministic synthetic dataset in MSLR-WEB10K SVMLight format
so every script and test runs in seconds, without the real download.

    python scripts/make_toy_data.py --out data/toy            # train.txt, vali.txt
    python scripts/make_toy_data.py --out data/toy_transfer --transfer-sets   # the three synthetic transfer sets

The toy data is NOT a benchmark: relevance is a noisy function of a latent
score, the four base-ranker channels (features 108, 110, 120, 125) are noisy
views of it, and a few other features carry a nonlinear signal a tree model can
use and plain fusion cannot.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness.feature_io import QueryData, write_svmlight_ltr
from harness.feature_names import BASE_RANKER_CHANNELS, N_FEATURES_MSLR

SPLITS = {"train": (120, 1), "vali": (40, 2)}  # name -> (n_queries, seed offset)


def make_split(n_queries: int, seed: int, docs_per_query: int = 25, n_features: int = N_FEATURES_MSLR, max_label: int = 4):
    rng = np.random.RandomState(seed)
    data = {}
    for q in range(n_queries):
        qid = str(1000 * seed + q + 1)
        z = rng.randn(docs_per_query)                         # latent relevance
        X = rng.randn(docs_per_query, n_features) * 0.5  # distractor features
        channels = BASE_RANKER_CHANNELS if n_features == N_FEATURES_MSLR else {f"c{j}": j + 1 for j in range(4)}
        for j, (name, idx) in enumerate(channels.items()):
            sig = 3 * z + (0.8 + 0.25 * j) * rng.randn(docs_per_query)   # noisy view of relevance
            # LMIR scores are log-probabilities: always <= 0, higher (closer to 0) is better
            X[:, idx - 1] = (-4 + 2 * np.tanh(sig / 4)) if name.startswith("lmir") else (5 + sig)
        a, b = (130, 131) if n_features == N_FEATURES_MSLR else (4, 5)
        X[:, a] = np.tanh(z * z) + 0.2 * rng.randn(docs_per_query)   # nonlinear signals
        X[:, b] = (z > 0.5) * 1.5 + 0.2 * rng.randn(docs_per_query)
        latent = z + 0.4 * rng.randn(docs_per_query) + 0.5 * (X[:, b] > 1)
        cuts = np.array([0.4, 1.0, 1.6, 2.2]) if max_label == 4 else np.array([0.7, 1.6])
        y = np.searchsorted(cuts, latent).astype(int)
        data[qid] = QueryData(qid=qid, doc_ids=[f"{qid}:{i}" for i in range(docs_per_query)],
                              labels=y.tolist(), features=X.astype(np.float32))
    return data


# Synthetic stand-ins for the undisclosed transfer benchmarks (Track C1): differing feature counts and label scales.
TRANSFER_SETS = {"a_40_features_labels0-4": (40, 4, 100), "b_90_features_labels0-4": (90, 4, 200),
                 "c_250_features_labels0-2": (250, 2, 300)}  # name -> (n_features, max_label, seed offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n-features", type=int, default=N_FEATURES_MSLR, help="!= 136 gives a non-MSLR layout")
    ap.add_argument("--max-label", type=int, choices=[2, 4], default=4)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--transfer-sets", action="store_true",
                    help="write the three synthetic transfer sets (40/90/250 features) under --out")
    ap.add_argument("--n-train", type=int, default=SPLITS["train"][0])
    ap.add_argument("--n-vali", type=int, default=SPLITS["vali"][0])
    args = ap.parse_args()
    sizes = {"train": args.n_train, "vali": args.n_vali}
    jobs = ([(os.path.join(args.out, name), nf, ml, off) for name, (nf, ml, off) in TRANSFER_SETS.items()]
            if args.transfer_sets else [(args.out, args.n_features, args.max_label, args.seed_offset)])
    for out, nf, ml, off in jobs:
        os.makedirs(out, exist_ok=True)
        for name, (_nq, seed) in SPLITS.items():
            nq = sizes[name]
            write_svmlight_ltr(os.path.join(out, f"{name}.txt"),
                               make_split(nq, seed + off, n_features=nf, max_label=ml))
            print(f"wrote {os.path.join(out, name + '.txt')}  ({nq} queries, {nf} features)")


if __name__ == "__main__":
    main()

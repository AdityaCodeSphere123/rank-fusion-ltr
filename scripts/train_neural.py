"""Train the pointwise and/or pairwise (RankNet) MLP scorers and save pointwise_model.pt / pairwise_model.pt.

    python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission

Calls train_model() from submission/pointwise_pairwise.py -- it will raise NotImplementedError until you have
written the TODOs. Early stopping on --dev counts as tuning against dev (it makes your dev score optimistic); declare that in your submission notes (Section 10).

--monitor train-slice keeps DEV unseen: one query fold of TRAIN (grouped_kfold, --monitor-folds folds, fold 0)
is held out as the early-stopping monitor and the model trains on the remaining TRAIN queries; --dev is then
only scored once, after training, as a reported number. --holdout-fold additionally removes a second TRAIN
fold from training and reports it, for choosing between settings without touching DEV.

Submitted models (see Decision.md):
    python scripts/train_neural.py --kind pointwise --train data/train --dev data/dev --monitor train-slice \
        --hidden 256 256 --out-dir submission
    python scripts/train_neural.py --kind pairwise --train data/train --dev data/dev --monitor train-slice \
        --pair-weighting delta_ndcg ... --out-dir submission      (exact flags: Decision.md, section 4)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json
import time

import numpy as np
import torch

from harness.cv import grouped_kfold
from scripts.data_cache import load_xyq, resolve
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model


def _resolve(path, name):
    return resolve(path, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default="data/toy")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pairs-per-query", type=int, default=300)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64],
                    help="hidden layer widths, e.g. --hidden 128 64 (at most 3 layers of at most 256 units)")
    ap.add_argument("--monitor", choices=["dev", "train-slice"], default="dev",
                    help="early-stopping split: the dev file (counts as tuning on DEV) or a TRAIN query fold")
    ap.add_argument("--monitor-folds", type=int, default=10)
    ap.add_argument("--holdout-fold", type=int, default=None,
                    help="also drop this TRAIN fold (of --monitor-folds) from training and report it")
    ap.add_argument("--pair-weighting", choices=["none", "delta_ndcg", "approx_ndcg"], default="none")
    ap.add_argument("--pair-sampling", choices=["uniform", "gain", "hard"], default="uniform",
                    help="how delta_ndcg draws its capped pairs: uniform, |gain| biased, or gain times current error")
    ap.add_argument("--approx-alpha", type=float, default=8.0, help="sharpness of the smooth ranks in approx_ndcg")
    ap.add_argument("--queries-per-batch", type=int, default=16)
    ap.add_argument("--ndcg-trunc", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-decay", type=float, default=1.0)
    ap.add_argument("--scale", choices=["standard", "robust"], default="standard",
                    help="affine feature scaler stored in the artifact (robust = median / IQR)")
    ap.add_argument("--use-dev", action="store_true",
                    help="append the dev split to the training rows. Only with --monitor train-slice; "
                         "the monitor and the holdout stay query-disjoint slices of TRAIN")
    ap.add_argument("--swa-start", type=int, default=0,
                    help="average weights from this epoch onward and keep them when the monitor is higher")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-save", action="store_true", help="experiment only: do not write the .pt artifact")
    ap.add_argument("--log", default=None, help="append a JSON summary line to this file")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    X, y, q = load_xyq(_resolve(args.train, "train.txt"))
    dev = load_xyq(_resolve(args.dev, "vali.txt"))
    holdout = None
    if args.monitor == "train-slice":
        folds = grouped_kfold(q, k=args.monitor_folds, seed=args.seed)
        mon_idx = folds[0][1]
        drop = set(mon_idx.tolist())
        if args.holdout_fold is not None:
            ho_idx = folds[args.holdout_fold][1]
            holdout = (X[ho_idx], y[ho_idx], q[ho_idx])
            drop |= set(ho_idx.tolist())
        keep = np.array([i for i in range(len(y)) if i not in drop])
        train = (X[keep], y[keep], q[keep])
        monitor = (X[mon_idx], y[mon_idx], q[mon_idx])
        if args.use_dev:
            Xd, yd, qd = dev
            train = (np.vstack([train[0], Xd]), np.concatenate([train[1], yd]), np.concatenate([train[2], qd]))
    else:
        if args.use_dev:
            raise SystemExit("--use-dev only applies with --monitor train-slice")
        train, monitor = (X, y, q), dev
    print(f"training rows {len(train[1])} ({len(set(train[2].tolist()))} queries); monitor = {args.monitor} "
          f"({len(set(monitor[2].tolist()))} queries)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for kind in (["pointwise", "pairwise"] if args.kind == "both" else [args.kind]):
        t0 = time.perf_counter()
        net, std, hist = train_model(kind, train, monitor, seed=args.seed, epochs=args.epochs,
                                     batch_size=args.batch_size, lr=args.lr, pairs_per_query=args.pairs_per_query,
                                     patience=args.patience, hidden=args.hidden, verbose=True,
                                     pair_weighting=args.pair_weighting, queries_per_batch=args.queries_per_batch,
                                     ndcg_trunc=args.ndcg_trunc, weight_decay=args.weight_decay,
                                     lr_decay=args.lr_decay, scale=args.scale,
                                     pair_sampling=args.pair_sampling, approx_alpha=args.approx_alpha,
                                     swa_start=args.swa_start)
        secs = time.perf_counter() - t0
        scorer = make_scorer(net, std)
        best = max(hist, key=lambda h: h["dev_ndcg@10"])
        summary = {"kind": kind, "args": vars(args), "epochs_run": len(hist), "best_epoch": best["epoch"],
                   "monitor_ndcg@10": best["dev_ndcg@10"], "dev_ndcg@10": dev_ndcg(scorer, *dev),
                   "train_seconds": round(secs, 1), "history": hist}
        if holdout is not None:
            summary["holdout_ndcg@10"] = dev_ndcg(scorer, *holdout)
        msg = (f"{kind}: epochs run = {len(hist)} (best {best['epoch']}), monitor nDCG@10 = {best['dev_ndcg@10']:.4f}, "
               f"DEV nDCG@10 = {summary['dev_ndcg@10']:.4f}")
        if holdout is not None:
            msg += f", TRAIN-holdout nDCG@10 = {summary['holdout_ndcg@10']:.4f}"
        if not args.no_save:
            path = os.path.join(args.out_dir, f"{kind}_model.pt")
            save_scorer(path, net, std, args.hidden)
            msg += f" -> {path}"
        print(msg + f"  ({secs:.0f}s)", flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()

"""Local dry run of Track C1. Calls submission/recipe.py train() on each synthetic transfer set in
data/toy_transfer/ (40 / 90 / 250 features, labels 0-4 or 0-2) and scores the resulting model on that set's vali
file through your own rerank_with_ltr(). The real hidden benchmarks are undisclosed and staff-only; these sets
only check that your recipe is dimension-agnostic, label-agnostic, fast and deterministic.

    python scripts/run_recipe.py                                   # all synthetic transfer sets
    python scripts/run_recipe.py --train my/train.txt --vali my/vali.txt   # any flat-feature pair you build
"""
import argparse
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness.recipe_eval import run_recipe_and_evaluate  # noqa: E402

CAP_SECONDS = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--vali", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.train:
        pairs = [(args.train, args.vali or args.train)]
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "toy_transfer")
        pairs = [(os.path.join(d, "train.txt"), os.path.join(d, "vali.txt")) for d in sorted(glob.glob(os.path.join(root, "*")))]
    for train, vali in pairs:
        with tempfile.TemporaryDirectory() as d:
            r = run_recipe_and_evaluate(train, vali, vali, os.path.join(d, "recipe_model.txt"), args.seed)
        flag = "  WARNING: over the 20-minute cap" if r["train_seconds"] > CAP_SECONDS else ""
        print(f"{os.path.basename(os.path.dirname(train)):<28} trained in {r['train_seconds']:5.1f}s (cap {CAP_SECONDS}s)  "
              f"vali nDCG@10 = {r['ndcg@10']:.4f}{flag}")


if __name__ == "__main__":
    main()

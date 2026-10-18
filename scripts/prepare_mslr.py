"""Copy MSLR-WEB10K Fold1 into the layout the repo expects. There is no scriptable download URL: get
MSLR-WEB10K.zip from Microsoft Research, unzip it, then:

    python scripts/prepare_mslr.py --fold-dir /path/to/MSLR-WEB10K/Fold1 --out data

Creates data/train/train.txt (Fold1 train.txt) and data/dev/vali.txt (Fold1 vali.txt). The Fold1 test.txt is the
withheld held-out split: staff keep it; do NOT copy it here (assignment Section 6).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for src, dst in (("train.txt", "train/train.txt"), ("vali.txt", "dev/vali.txt")):
        s = os.path.join(args.fold_dir, src)
        if not os.path.isfile(s):
            raise SystemExit(f"missing {s}")
        d = os.path.join(args.out, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        print(f"{s} -> {d}")


if __name__ == "__main__":
    main()

"""Parse an SVMLight LTR file once with harness.feature_io and cache (X, y, qids) as .npz OUTSIDE the repo.

Parsing the 838 MB MSLR TRAIN file with the pure-Python reader takes minutes; every experiment script here reuses
the cache instead. The cache lives in $A3_CACHE_DIR (default: <system temp>/a3_ltr_cache) so no data file ever
lands in the submission tree. The cache key is the file's absolute path + size + mtime.

    python scripts/data_cache.py data/train/train.txt data/dev/vali.txt     # warm the cache
"""
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.feature_io import read_svmlight_ltr, stack


def cache_dir() -> str:
    d = os.environ.get("A3_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "a3_ltr_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def resolve(path: str, name: str) -> str:
    return os.path.join(path, name) if os.path.isdir(path) else path


def load_xyq(path: str):
    """(X float32, y int64, qids object array) for an LTR file, cached."""
    cpath = os.path.join(cache_dir(), f"{os.path.basename(path)}.{_key(path)}.npz")
    if os.path.isfile(cpath):
        z = np.load(cpath, allow_pickle=False)
        return z["X"], z["y"], z["q"].astype(object)
    X, y, q = stack(read_svmlight_ltr(path))
    np.savez(cpath, X=X, y=y, q=q.astype(str))
    return X, y, q


if __name__ == "__main__":
    import time
    for p in sys.argv[1:]:
        t0 = time.perf_counter()
        X, y, q = load_xyq(p)
        print(f"{p}: X={X.shape} queries={len(set(q.tolist()))} labels={np.bincount(y).tolist()} "
              f"({time.perf_counter() - t0:.1f}s)")

"""
harness/feature_io.py -- reading learning-to-rank data in the SVMLight/LETOR
format that MSLR-WEB10K and other LTR benchmarks ship in:

    <label> qid:<qid> 1:<v1> 2:<v2> ... N:<vN> [# comment]

One line per query-document pair; all lines of a query are contiguous. These
files carry NO document identifiers, so this reader synthesizes one per row:
doc_id = "<qid>:<row index within that query's block>". The synthetic id is
stable for a given file, which is all the harness needs.

Dependency-free (numpy only). Sparse lines (missing feature indices) are
filled with 0.0, matching the format's convention.
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


class QueryData:
    __slots__ = ("qid", "doc_ids", "labels", "features")

    def __init__(self, qid: str, doc_ids: List[str], labels: np.ndarray, features: np.ndarray):
        self.qid = qid
        self.doc_ids = doc_ids
        self.labels = labels          # int array, shape (n_docs,)
        self.features = features      # float32 array, shape (n_docs, n_features)


Dataset = "OrderedDict[str, QueryData]"


def read_svmlight_ltr(path: str, n_features: Optional[int] = None) -> "OrderedDict[str, QueryData]":
    """Parse an LTR file into {qid: QueryData}, preserving file order.
    n_features=None infers the width from the largest feature index seen."""
    rows_by_qid: "OrderedDict[str, list]" = OrderedDict()
    max_index = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            try:
                label = int(float(parts[0]))
                if not parts[1].startswith("qid:"):
                    raise ValueError("second token must be qid:<id>")
                qid = parts[1][4:]
                feats = []
                for tok in parts[2:]:
                    idx, val = tok.split(":")
                    idx = int(idx)
                    feats.append((idx, float(val)))
                    if idx > max_index:
                        max_index = idx
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no}: malformed LTR line ({e}): {line[:80]!r}") from e
            rows_by_qid.setdefault(qid, []).append((label, feats))

    width = n_features if n_features is not None else max_index
    if max_index > width:
        raise ValueError(f"{path}: feature index {max_index} exceeds n_features={width}")

    data: "OrderedDict[str, QueryData]" = OrderedDict()
    for qid, rows in rows_by_qid.items():
        n = len(rows)
        X = np.zeros((n, width), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for r, (label, feats) in enumerate(rows):
            y[r] = label
            for idx, val in feats:
                X[r, idx - 1] = val
        data[qid] = QueryData(qid, [f"{qid}:{r}" for r in range(n)], y, X)
    return data


def stack(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a dataset to (X, y, qids): X (rows, n_features) float32,
    y (rows,) int labels, qids (rows,) array of query-id strings, all in
    file order (queries stay contiguous)."""
    Xs, ys, qs = [], [], []
    for qid, q in dataset.items():
        Xs.append(q.features)
        ys.append(q.labels)
        qs.extend([qid] * len(q.labels))
    if not Xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.int64), np.array([], dtype=object)
    return np.vstack(Xs), np.concatenate(ys), np.array(qs, dtype=object)


def to_qrels(dataset) -> Dict[str, Dict[str, int]]:
    """{qid: {doc_id: graded label}} for harness.metrics."""
    return {qid: {d: int(l) for d, l in zip(q.doc_ids, q.labels)} for qid, q in dataset.items()}


def write_svmlight_ltr(path: str, dataset) -> None:
    """Inverse of read_svmlight_ltr (used by scripts/make_toy_data.py)."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, q in dataset.items():
            for label, row in zip(q.labels, q.features):
                feats = " ".join(f"{i + 1}:{float(v):.6g}" for i, v in enumerate(row))
                f.write(f"{int(label)} qid:{qid} {feats}\n")

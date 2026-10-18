# Submission interface (assignment Sections 4.2, 5)

`submission/ranking.py` must expose, with exactly these parameter names:

| function | returns |
|---|---|
| `prepare(model_path: str) -> None` | loads the trained artifact. **Must raise** if the file is missing or unreadable -- never train, never fall back to an untrained model. |
| `fuse_candidates(qid, candidate_doc_ids, base_scores, k)` | `List[(doc_id, score)]`, best first, at most `k` |
| `rerank_with_ltr(qid, candidate_doc_ids, features, k)` | same |

* `candidate_doc_ids`: the query's whole candidate pool (MSLR has ~120 docs/query). MSLR files carry **no document ids**, so the harness synthesizes `"<qid>:<row-within-query>"`. Treat ids as opaque strings.
* `base_scores[doc_id]` = `{"bm25_whole_document": x, "bm25_title": x, "lmir_dir_whole_document": x, "lmir_jm_whole_document": x}`. Higher is better for every channel (LMIR values are negative log-probabilities; larger = better). These are MSLR features 110, 108, 120, 125 (1-based; `harness/feature_names.py`).
* `features[doc_id]` = the full 136-float vector, index `i` = MSLR feature `i+1`.
* Return only ids from `candidate_doc_ids`, no duplicates, at most `k`. The harness re-sorts by score and rejects violations.
* Ties: break deterministically (the baseline keeps candidate order). `rerank_with_ltr` must be deterministic.

## Fusion conventions (`submission/fusion.py`)
* `rrf_scores` (you write it): ranks are 1-based within each channel; equal scores are ordered by candidate order (stable sort).
* CombSUM/CombMNZ: min-max normalise **each channel over the candidate pool**; a constant channel normalises to all zeros.
* CombMNZ: "appears in the list" is defined as *normalised score > 0* (every candidate has a raw score in every channel, so presence is otherwise always true).

## Artifacts to ship in `submission/`
`model.txt` (LambdaMART in LightGBM text format; the path the harness passes to `prepare`), plus `pointwise_model.pt` and `pairwise_model.pt`, written with `save_scorer` and loaded by staff with their own copy of `harness/neural.py` (`load_scorer`; plain MLP, at most 3 hidden layers of at most 256 units). Both feed the "learns something" check (Section 4.3 test (e)); `pairwise_model.pt` is also scored on held-out data as Track P. Also `submission/recipe.py` (Track C1; contract in docs/TRANSFER_TRACK.md).

Functions staff test by name: `rrf_scores`, `combsum_scores`, `combmnz_scores` (fusion.py); `set_seed`, `pointwise_loss`, `make_pairs`, `ranknet_loss`, `pointwise_step`, `pairwise_step`, `train_model`, `load_scorer` (pointwise_pairwise.py).

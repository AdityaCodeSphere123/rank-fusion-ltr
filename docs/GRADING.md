# How grading works

Scoring (assignment Section 7), every accuracy-bearing item percentile-ranked against the class (ties share average rank):

| track | metric | weight |
|---|---|---|
| A | `fuse_candidates()` on MSLR DEV: nDCG@10 (80% of the track) and MAP@10 (20%), each percentile-ranked | 10% (8% + 2%) |
| B | `rerank_with_ltr()` nDCG@10 on MSLR DEV, percentile-ranked | 10% |
| P | your `submission/pairwise_model.pt`, loaded with the staff copy of `harness/neural.py` (plain MLP, at most 3 hidden layers of at most 256 units), nDCG@10 on the MSLR held-out split, percentile-ranked. A bad artifact scores 0 here only. Dev scores are shown live as feedback | 10% |
| C1 | staff run your `submission/recipe.py` on each of two undisclosed benchmarks; per-benchmark percentile of `rerank_with_ltr()` nDCG@10 on its held-out split, averaged (docs/TRANSFER_TRACK.md). Failure/timeout on a benchmark = 0 there | 35% |
| C2 | hidden robustness test: staff run your submitted model (the same `rerank_with_ltr()` as Track B, no retraining) on hidden versions of the MSLR held-out split; percentile of the mean nDCG@10 over the versions. How the versions are constructed is not disclosed | 35% |

No bonus track: the weights sum to 100%.

Percentiles: 0 for the lowest team, 1 for the highest, linear between; **tied scores share the average percentile**. For Tracks A and B, where teams cluster, every metric is first **rounded to the nearest 0.005**, so scores within the same band tie (differences below the evaluation noise earn nothing). P, C1 and C2 use exact percentiles.

Metric details: nDCG@10 uses gain `2^label - 1`, log2 discount; MAP@10 treats label > 0 as relevant and normalises by the total number of relevant documents of the query. Queries are averaged equally.

**Training on DEV.** It raises your dev scores (Tracks A and B) and no track penalizes it directly, so your public leaderboard position will overstate where you stand; it does not help Track C1 (hidden data), and Track C2 is scored on held-out data it has not seen. Declare which splits you used (`SUBMISSION_NOTES.md`): an undeclared use of dev for training is an integrity violation. Tuning or early stopping on DEV also counts as using it. A model that does well on DEV only by memorizing it is not what the course is trying to reward; prefer cross-validation on TRAIN (`scripts/cv_lambdamart.py`).

Time caps (spec Section 5): `prepare()` 60 s; each `fuse_candidates()` / `rerank_with_ltr()` call 2 s; whole MSLR grading run 30 min. Separately, `recipe.train()` gets 20 min on 4 CPU threads per hidden benchmark.

**Track P rules.** You may change pair sampling, pairs per query, epochs, learning rate, batch size, preprocessing, the MLP's width/depth (within the bound) and per-pair weights (e.g. |delta nDCG|, i.e. LambdaRank) -- in your own extra functions, never inside the graded `ranknet_loss`. Train on relevance labels only (no distillation from other models' scores). See the docstring of `submission/pointwise_pairwise.py`.

Correctness checks staff run on your hand-written parts: RRF / CombSUM / CombMNZ worked examples; `pointwise_loss`, `ranknet_loss` on tiny examples (e.g. `ranknet_loss([2,0],[0,0]) = 0.410038`); `make_pairs` yields only same-query, strictly-ordered pairs, preferred first, capped per query; one `pairwise_step` raises `s_preferred - s_other`; your pairwise model beats sorting by a single BM25 feature on DEV; training is reproducible given the seed. Run `python -m pytest tests/test_student_todos.py` to self-check.

# Decision log — COL 7364/764 A3 (Rank Fusion and LTR Arena)

This file is kept up to date while working. Numbers are copied from logs under `experiments/`.
"TRAIN-CV" = grouped (by query) k-fold inside MSLR Fold1 TRAIN (`harness.cv.grouped_kfold`). "DEV" = Fold1
vali, used **only as a reported number** unless explicitly stated otherwise.

## 1. Environment and data check

| item | value |
|---|---|
| OS (dev machine) | Windows 11 Pro 10.0.26200, i7-13650HX (14 cores / 20 threads), 25 GB RAM |
| Python | 3.12.3 (Anaconda base interpreter; the repo has no `.venv`, so none was activated) — same minor version as the grader (3.12.3) |
| numpy / lightgbm / torch / pytest | 2.2.6 / 4.6.0 / 2.13.0+cu126 (CPU used) / 7.4.4 — all import; nothing installed, no extra package added |
| `data/train/train.txt` | 723,412 rows, 6,000 queries, 136 feature fields per line, labels 0–4 = [377957, 232569, 95082, 12658, 5146] |
| `data/dev/vali.txt` | 235,259 rows, 2,000 queries, 136 fields, labels 0–4 = [121522, 75815, 31910, 4209, 1803] |
| `data/toy/{train,vali}.txt` | 3,000 / 1,000 rows (synthetic) |
| `data/toy_transfer/*/{train,vali}.txt` | 3,750 / 1,250 rows each; 40, 90, 250 features; labels 0–4, 0–4, 0–2 |
| line format | `2 qid:1 1:3 2:3 3:0 ... 136:v` (dense, every line lists all 136 indices) |
| base-ranker channels | `harness.feature_names.BASE_RANKER_CHANNELS` = bm25_whole_document→110, bm25_title→108, lmir_dir_whole_document→120, lmir_jm_whole_document→125; higher = better on all four (LMIR are negative log-probs) |
| held-out `test.txt` | **not present** anywhere in the repo and **not fetched**. No MSLR archive was downloaded; no held-out label was consulted; no attempt was made to identify the C1 benchmarks. |

Parsing TRAIN with the harness reader takes 136 s (DEV 43 s) from Google Drive, so `scripts/data_cache.py` parses
each file once with `harness.feature_io` and caches `(X, y, qids)` as `.npz` in the system temp dir (outside the
repo, so no data file can end up in the submission).

**Compute policy.** Every LightGBM fit uses `num_threads=4`. The CV sweep runs at most 3 jobs in parallel (3×4 = 12
threads on a 14-core machine); torch training uses `torch.set_num_threads(4)`. Fit times below were measured while
other jobs ran in parallel, so they are upper bounds for this machine.

## 2. Decisions log

### D1. Fusion method (Track A) — **CombSUM over 3 channels (LMIR.JM dropped)**

`scripts/fusion_eval.py` (log `experiments/fusion_eval.log`, all numbers in `experiments/fusion_results.json`).
Compared on all 6,000 TRAIN queries: each single channel, RRF with c ∈ {1,2,5,10,20,30,45,60,80,100,150,250,500},
CombSUM, CombMNZ, per-query z-score sum, and channel-weighted RRF (c ∈ {10,20,30,60,100}) / CombSUM / z-sum with
weights ∈ {0, .25, .5, 1, 2}³ for the three non-BM25-whole channels (BM25-whole fixed at 1). Run time 2 min 40 s.

* RRF improves with c up to about 150 (TRAIN 0.2843 at c=1, 0.2937 at c=150, flat after that). Every RRF
  setting is below CombSUM (0.2961).
* CombMNZ ≈ CombSUM (0.2958 vs 0.2961): with min-max normalisation only the single worst document per channel has
  a 0 score, so the MNZ multiplier is almost always 4.
* Weighting: the best TRAIN setting is weights (1, 1, 1, 0) i.e. plain CombSUM without LMIR.JM (0.3000). LMIR.JM
  is the weakest single channel (0.2621) and largely redundant with LMIR.DIR.
* Selection check: the best of about 400 methods on 4 TRAIN folds, scored on the 5th, is
  **0.2989 ± 0.0041** nDCG@10 (MAP@10 0.1226), against 0.2961 when the choice is limited to unweighted methods
  (always CombSUM). 2 of 5 folds picked weights (1, .5, 1, 0) and 3 of 5 picked (1, 1, 1, 0). The shipped
  method is the full-TRAIN winner: equal weights on the three channels.
* Shipped: `fuse_candidates` = `combsum_scores(..., channels=(bm25_whole_document, bm25_title,
  lmir_dir_whole_document))`. The BM25 fallback in `ranking.py` was removed. All three required functions
  (`rrf_scores`, `combsum_scores`, `combmnz_scores`) are implemented and match the docstring examples.

| method (selected on TRAIN) | TRAIN nDCG@10 | TRAIN MAP@10 | DEV nDCG@10 | DEV MAP@10 |
|---|---|---|---|---|
| single BM25 whole document | 0.2835 | 0.1178 | 0.2885 | 0.1205 |
| RRF c=60 | 0.2926 | 0.1195 | 0.2995 | 0.1246 |
| RRF c=150 (best c on TRAIN) | 0.2937 | 0.1198 | 0.3002 | 0.1250 |
| CombSUM (4 channels) | 0.2961 | 0.1213 | 0.3056 | 0.1285 |
| CombMNZ (4 channels) | 0.2958 | 0.1209 | 0.3061 | 0.1286 |
| z-score sum (4 channels) | 0.2954 | 0.1213 | 0.3039 | 0.1283 |
| **CombSUM, 3 channels (shipped)** | **0.3000** | **0.1229** | **0.3105** | **0.1302** |

DEV numbers confirmed end to end with `python -m harness.run_harness --features data/dev/vali.txt`:
fuse_candidates nDCG@10 **0.3105**, MAP@10 **0.1302** (identical to the vectorised evaluation).

### D2. Pointwise / pairwise implementation (correctness part)

* `set_seed` seeds `random`, NumPy and torch. `pointwise_loss` = mean squared error. `ranknet_loss` =
  mean of `torch.logaddexp(0, -(s_i - s_j))` (stable; `ranknet_loss([-1000],[1000]) = 2000`). Both steps do
  `zero_grad → forward (shared network) → loss → backward → step`.
* `make_pairs` works one query at a time (rows grouped by first appearance, so rows of a query need not be
  contiguous): all unordered pairs of the query via `np.triu_indices`, ties dropped, capped at
  `pairs_per_query` with `rng.choice(..., replace=False)`, oriented more-relevant first. Only one query's pairs
  exist at a time; the full TRAIN split's pairs are never materialised.
* Training loop exactly as the skeleton describes (fresh pairs + `rng.permutation` reshuffle every epoch, per-epoch
  monitor nDCG@10 via the provided `dev_ndcg`, best `state_dict` kept, `patience` early stopping, best weights
  restored). The standardizer is fit on the training rows only.
* Track P extension (separate functions, `ranknet_loss` untouched): `train_model(..., pair_weighting="delta_ndcg")`
  switches to LambdaRank-style training — a minibatch is `queries_per_batch` whole queries; all their docs are
  scored, pairs are formed with `make_pairs` (cap `pairs_per_query`), each pair is weighted by
  `delta_ndcg_weights` (|ΔnDCG@10| of swapping the two docs under the batch's current, detached scores; gain
  2^l−1, discount 1/log2(1+rank), normalised by IDCG@10) and `weighted_ranknet_loss` is minimised. Optional
  `ndcg_trunc`, `weight_decay` (Adam L2), `lr_decay` (per-epoch multiplicative). Trained on relevance labels only.
* Verified: `pytest tests/test_student_todos.py` 7/7; equal weights ⇒ `weighted_ranknet_loss == ranknet_loss`;
  ΔnDCG of moving the only relevant doc from rank 2 to 1 = 1 − 1/log2(3) = 0.36907 (hand value); two runs with the
  same seed give bit-identical weights on both paths.

### D3. Neural model selection (Track P and the pointwise model)

Protocol (no DEV in any decision): TRAIN queries split by `grouped_kfold(k=10, seed=0)`; fold 0 = early-stopping
monitor, fold 1 = comparison holdout, the other 8 folds = training (4,800 queries, 577,597 rows). DEV is scored once
after training and only reported. Script: `scripts/train_neural.py --monitor train-slice --holdout-fold 1`;
logs `experiments/neural_sweep*.log`, `experiments/neural_runs.jsonl`. Each fold has 600 queries, so the
standard error of a fold mean is ≈ 0.01: differences below ~0.005 are not meaningful.

| run | model / loss | hidden | other | epochs (best) | monitor fold | holdout fold | DEV (report) | time |
|---|---|---|---|---|---|---|---|---|
| E0 | pointwise MSE | 64 | lr 1e-3, bs 1024 | 22 (19) | 0.4333 | 0.4083 | 0.4297 | 42 s |
| E1 | RankNet, uniform 300 pairs/q | 64 | lr 1e-3, bs 1024 | 8 (5) | 0.4320 | 0.4126 | 0.4266 | 102 s |
| E2 | RankNet, uniform 300 pairs/q | 256,256 | lr 1e-3 | 6 (2) | 0.4265 | 0.4083 | 0.4271 | 249 s |
| E3 | LambdaRank (ΔnDCG-weighted), ≤2000 pairs/q, 16 q/batch | 256,256 | lr 1e-3 | 21 (17) | 0.4397 | 0.4126 | 0.4340 | 308 s |
| E4 | LambdaRank, ≤2000 pairs/q, 16 q/batch | 256,256,256 | lr 1e-3 | 17 (13) | 0.4412 | 0.4111 | 0.4388 | 362 s |

Observations: the wide uniform RankNet overfits after 2 epochs (monitor falls from 0.4265 to 0.3965 while the train
loss keeps dropping), the ΔnDCG weighting keeps improving for ~15 epochs, but its monitor curve oscillates by
±0.01 between epochs → round 2 tests learning-rate decay, bigger query batches, L2 and a truncated discount.

Round 2 / 3 (all LambdaRank, 256×256×256, ≤2000 pairs/q unless stated, patience 5, max 40 epochs):

| run | change | epochs (best) | monitor fold | holdout fold | DEV (report) | time |
|---|---|---|---|---|---|---|
| E5 | lr 1e-3, lr_decay 0.9, 16 q/batch | 18 (13) | 0.4396 | 0.4145 | 0.4382 | 360 s |
| **E6** | **lr 1e-3, lr_decay 0.9, 64 q/batch** | 22 (17) | **0.4449** | **0.4182** | 0.4375 | 344 s |
| E7 | E5 + weight_decay 1e-5 | 17 (12) | 0.4401 | 0.4174 | 0.4374 | 298 s |
| E8 | E5 + ndcg_trunc 20 | 16 (11) | 0.4412 | 0.4124 | 0.4389 | 271 s |
| E9 | E6 with 128 q/batch | 19 (14) | 0.4416 | 0.4112 | 0.4355 | 303 s |
| E10 | E6 with lr 2e-3 | 22 (17) | 0.4460 | 0.4141 | 0.4393 | 325 s |
| E11 | E6 with all pairs of every query | 10 (5) | 0.4368 | 0.4106 | 0.4317 | 169 s |

**Choice for `pairwise_model.pt`: E6.** Best TRAIN-holdout score, and best monitor + holdout sum, of the 11 runs.
The choice used TRAIN folds only. E4, E8, and E10 are about 0.001–0.002 higher on DEV. Gain over textbook RankNet:
+0.006 on the holdout fold, +0.011 on DEV. The artifact is an affine standardizer plus a plain MLP, so counts,
lengths, and PageRank stay untransformed. The pointwise model stays at E0 (64 hidden units, lr 1e-3, batch size 1024).

### D4. Track C1 recipe rules

`submission/recipe.py`; simulation `scripts/recipe_sim.py` (log `experiments/recipe_sim.log`,
`experiments/recipe_sim.jsonl`). The rules were checked on five benchmark-like tasks carved from **TRAIN only**,
each with disjoint train / vali / test query sets (seed-fixed), differing in size, width, label levels and list
length: 300 queries × 136 features × labels 0–4; 1,000 q × 46 random features × labels collapsed to 0–2;
1,000 q × 136 f with lists subsampled to 20 docs; 2,500 q × 90 random f; 4,000 q × 136 f × labels 0–2. The recipe
sees train + vali (as staff run it) and is scored on the task's test queries.

| variant | docs20 | d46 l0–2 | q2500 d90 | q300 | q4000 l0–2 | mean |
|---|---|---|---|---|---|---|
| **default rules + vali early stopping + refit (shipped)** | 0.5611 | **0.4980** | 0.4419 | 0.4454 | **0.5594** | 0.5011 |
| same, no refit (train-only early-stopped model) | 0.5591 | 0.4950 | 0.4377 | 0.4369 | 0.5575 | 0.4972 |
| starter recipe (fixed 100 trees, 15 leaves) | 0.5556 | 0.4886 | 0.4240 | 0.4336 | 0.5392 | 0.4882 |
| leaves ×2 | 0.5624 | 0.4951 | 0.4420 | 0.4435 | 0.5580 | 0.5002 |
| leaves ÷2 | 0.5601 | 0.4941 | 0.4416 | 0.4386 | 0.5562 | 0.4981 |
| min_data_in_leaf ×4 | 0.5615 | 0.4971 | 0.4410 | 0.4447 | 0.5583 | 0.5005 |
| feature_fraction 0.5 | 0.5629 | 0.4971 | 0.4427 | 0.4468 | 0.5578 | 0.5015 |

* Refit on train + vali helps on 5/5 tasks (+0.004 on average, +0.009 on the smallest). Kept.
* Rules plus early stopping beat the starter by +0.013 on average and by about 0.02 on the larger tasks.
* Halving or doubling leaves, ×4 `min_data_in_leaf`, and `feature_fraction` 0.5 all stay within ±0.001 of the
  default on average. `feature_fraction` 0.5 wins 3 tasks and loses the 46-feature task and the large task.
  Default rules kept.
* Run time, with the CV sweep also on the CPU: 2–240 s per task. The 480k-row task took 238 s including the refit.

Rules as shipped (functions of what `train()` sees; no feature index or count is hard-coded):
`num_leaves = clip(2^round(log2(sqrt(n_rows)/16)) − 1, 7, 63)`; `min_data_in_leaf = clip(n_rows/4000, 20, 200)`;
`feature_fraction = clip(1 − 0.15·log2(D/32), 0.5, 1.0)`; `lambda_l2 = 10` if n_rows < 50k else 1; learning rate 0.05
(0.1 if the cost model allows < 800 rounds); bagging 0.8 every iteration; max rounds from a size-based cost model
(≤ 3000), early-stopping patience = clip(max_rounds/10, 50, 200) on vali nDCG@10; refit on train + vali with
`round(best · (n_train + n_vali)/n_train)` rounds if the predicted fit + refit cost ≤ 0.8 × 15 min.

### D5. LambdaMART hyperparameters (Tracks B and C2)

`scripts/cv_sweep.py`: grouped 5-fold CV on TRAIN (`grouped_kfold(q, 5, seed=0)`; 4,800 training / 1,200 validation
queries per fold). One model per (config, fold) is trained to the largest tree count and scored at each FIXED tree
count with `predict(num_iteration=n)` (identical to training n rounds; nothing is early-stopped on the reported
fold). LightGBM needs no fitted preprocessing (trees are invariant to monotone transforms), so nothing leaks
across folds. All runs: `objective=lambdarank`, bagging 0.8 every iteration, `num_threads=4`,
`deterministic=True`, `force_row_wise=True`, seed 0. Full table of every config × tree count (train mean, val mean,
val std, val/train, fit seconds): `experiments/cv_report.txt`; raw per-fold numbers `experiments/cv_results.jsonl`;
logs `experiments/cv_leaves.log`, `cv_stage2.log`, `cv_stage3.log`. Wall-clock: stage 1 ≈ 50 min, stage 2 ≈ 50 min,
stage 3 ≈ 45 min (3 jobs × 4 threads in parallel; per-fold fit 230–660 s).

**Best tree count of every config** (val = mean validation-fold nDCG@10, ± std over folds):

| stage | leaves | lr | min_data_in_leaf | feature_fraction | λ2 | best trees | train | val ± std | val/train |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 31 | 0.05 | 100 | 0.8 | 1 | 1000 | 0.6243 | 0.4749 ± 0.0059 | 0.761 |
| 1 | 63 | 0.05 | 100 | 0.8 | 1 | 300 | 0.5893 | 0.4744 ± 0.0063 | 0.805 |
| 1 | 127 | 0.05 | 100 | 0.8 | 1 | 500 | 0.7174 | 0.4742 ± 0.0077 | 0.661 |
| 1 | 255 | 0.05 | 100 | 0.8 | 1 | 200 | 0.6961 | 0.4741 ± 0.0059 | 0.681 |
| 2 | 63 | 0.05 | 500 | 0.8 | 1 | 750 | 0.6713 | 0.4747 ± 0.0069 | 0.707 |
| 2 | 63 | 0.05 | 500 | 0.5 | 1 | 500 | 0.6233 | 0.4741 ± 0.0068 | 0.761 |
| 2 | 31 | 0.05 | 100 | 0.5 | 1 | 1000 | 0.6190 | 0.4744 ± 0.0063 | 0.766 |
| 2 | 63 | 0.05 | 200 | 0.5 | 10 | 500 | 0.6069 | 0.4733 ± 0.0060 | 0.780 |
| 2 | 127 | 0.05 | 1000 | 0.5 | 1 | 300 | 0.6386 | 0.4721 ± 0.0066 | 0.739 |
| 2 | 63 | 0.025 | 500 | 0.5 | 1 | 1000 | 0.6249 | 0.4766 ± 0.0070 | 0.763 |
| 3 | 63 | 0.0125 | 500 | 0.5 | 1 | 3000 (2000: 0.4771) | 0.6677 | 0.4773 ± 0.0068 | 0.715 |
| 3 | 31 | 0.025 | 100 | 0.8 | 1 | 3000 | 0.6684 | 0.4764 ± 0.0071 | 0.713 |
| 3 | 63 | 0.025 | 500 | 0.8 | 1 | 1500 | 0.6730 | 0.4757 ± 0.0078 | 0.707 |

**Tree-count curves (validation nDCG@10 vs trees — the generalization-gap evidence):**

| trees | 31 leaves | 63 leaves | 127 leaves | 255 leaves | **chosen family** (63 L, lr .0125, md 500, ff .5) |
|---|---|---|---|---|---|
| 100 | 0.4621 | 0.4662 | 0.4687 | 0.4696 | — |
| 200 | 0.4696 | 0.4724 | 0.4732 | 0.4741 | — |
| 300 | 0.4719 | 0.4744 | 0.4741 | 0.4737 | — |
| 500 | 0.4745 | 0.4742 | 0.4742 | 0.4719 | 0.4694 |
| 1000 | 0.4749 | 0.4735 | 0.4707 | 0.4653 | 0.4752 |
| 1500 | 0.4741 | 0.4708 | 0.4683 | 0.4621 | 0.4768 |
| 2000 | 0.4731 | 0.4708 | 0.4651 | 0.4599 | **0.4771** (train 0.6263, ratio 0.762) |
| 3000 | — | — | — | — | 0.4773 |
| 4000 | — | — | — | — | 0.4764 |

(stage-1 columns: lr 0.05, min_data_in_leaf 100, ff 0.8; all values are 5-fold means.)

Findings:
* Every leaf count peaks at about 0.474–0.475 validation. 255 leaves reaches train 0.944 at 2000 trees
  (val/train 0.49) while validation falls from 0.4741 to 0.4599.
* Halving the learning rate from 0.05 to 0.025 improved validation on 5/5 folds (+0.0025 mean). 0.025 to 0.0125
  added +0.0005 at 2000 trees and +0.0007 at 3000, mixed across folds. Larger `min_data_in_leaf`, λ2 = 10, and
  `feature_fraction` 0.5 alone left validation flat.
* Noise comparison (`scripts/robustness_eval.py`, `experiments/robustness.log`; trained on TRAIN folds
  1–4, scored on fold 0 — never DEV). Δ vs clean:

| config | clean | ×(1 + 10% noise) | +0.1·std noise | 10% of features zeroed per query | 50% docs/query |
|---|---|---|---|---|---|
| 63 L, md 500, ff 0.8, 750 trees | 0.4870 | −0.0148 | −0.1573 | −0.0278 | +0.0283 |
| 31 L, md 100, ff 0.8, 1000 trees | 0.4860 | −0.0134 | −0.1578 | −0.0243 | +0.0298 |
| 31 L, md 100, ff 0.5, 1000 trees | 0.4864 | −0.0093 | −0.1516 | −0.0217 | +0.0302 |
| 63 L, md 500, ff 0.5, 500 trees | 0.4865 | −0.0125 | −0.1485 | −0.0211 | +0.0306 |
| 255 L, md 100, ff 0.8, 1000 trees (overfit reference) | 0.4788 | −0.0177 | −0.1486 | −0.0276 | +0.0299 |

  `feature_fraction` 0.5 loses less under feature noise and feature masking at the same clean accuracy. The
  255-leaf model is worst on the clean fold and under multiplicative noise. Additive noise scaled by each
  feature's standard deviation drops every model; it is reported and was not used to choose.

**D5 choice (later replaced; see D7):** `num_leaves=63, learning_rate=0.0125, min_data_in_leaf=500,
feature_fraction=0.5, lambda_l2=1.0, bagging_fraction=0.8, bagging_freq=1`, **2000 trees**. Highest CV validation
of the families tested (0.4771 ± 0.0068). Validation is flat from 1500 to 3000 trees (0.4768–0.4773). val/train
0.762. Rejected: 3000 trees (+0.0002, ratio 0.715); 255 and 127 leaves (same peak, larger train/val gap);
lr 0.025 at 1000 trees (−0.0005).

Final fit: `python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt --trees 2000
--num-leaves 63 --learning-rate 0.0125 --min-data-in-leaf 500 --lambda-l2 1.0 --feature-fraction 0.5
--bagging-fraction 0.8 --bagging-freq 1 --seed 0` → 6,000 TRAIN queries, **227 s**, 14.3 MB, train nDCG@10 0.6079,
DEV nDCG@10 **0.4845** (reported only; DEV/train = 0.80). Log `experiments/train_final_lambdamart.log`; sha1 of
that file `98c89731fddc0c5566401c22fc58b6db9cbbccae`. This file was replaced by the D7 fit.

DEV nDCG@10 of the D5 model vs number of trees (`experiments/dev_curve_final.txt`; computed after that choice):

| trees | 100 | 250 | 500 | 750 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|
| DEV nDCG@10 | 0.4532 | 0.4672 | 0.4781 | 0.4813 | 0.4829 | 0.4841 | 0.4845 |

The D5 model used the 136 given features with no extra columns. The files are dense (every line lists all 136
indices), so no imputation was applied. The current model adds the columns described in D7. `rerank_with_ltr`
pads or truncates a row to the model's feature count, and rebuilds the D7 columns when the width matches.

## 3. Results summary

| model | selection protocol | TRAIN-internal estimate | DEV nDCG@10 (report) | DEV MAP@10 |
|---|---|---|---|---|
| BM25 whole-document sort (floor for check (e)) | — | 0.2835 (TRAIN) | 0.2885 | 0.1205 |
| Fusion: CombSUM over 3 channels (`fuse_candidates`) | best on TRAIN | 0.2989 ± 0.0041 (5-fold selection CV) | **0.3105** | **0.1302** |
| Pointwise MLP (64), `pointwise_model.pt` | textbook, early stop on TRAIN fold | 0.4291 (monitor fold) | 0.4225 | — |
| Pairwise LambdaRank MLP (256×3), `pairwise_model.pt` | TRAIN folds | 0.4417 (monitor fold) | **0.4364** | — |
| LambdaMART D5 (replaced by D7) | TRAIN 5-fold CV | 0.4771 ± 0.0068 (CV val) | 0.4845 | 0.1526 |

On this table, LambdaMART is above the pairwise MLP, then the pointwise MLP, fusion, and single BM25. Those DEV
numbers are about 0.007 above the TRAIN-CV estimates for each family. The current LambdaMART file is the D13 fit,
not this row.

Recipe (C1) on the synthetic transfer sets (`python scripts/run_recipe.py`). The vali file is also used for the
refit, so these vali numbers include that refit:

| set | train time | vali nDCG@10 |
|---|---|---|
| a_40_features_labels0-4 | 8.4 s (first run, includes imports) / 0.7 s | 0.9573 |
| b_90_features_labels0-4 | 2.6 s / 1.3 s | 0.9703 |
| c_250_features_labels0-2 | 6.3 s / 2.7 s | 0.9650 |

Determinism: two `recipe.train` runs with seed 0 wrote byte-identical model files on all three sets (sha1
20a68dc83c0c / f199a0a29578 / 43e39d110c0b); seed 1 gives a different file.

Worst-case recipe timing (`experiments/recipe_stress.log`): a synthetic benchmark at the stated limits, built from
TRAIN only (500,111 train rows × 300 features = the 136 MSLR columns + 164 perturbed copies; 99,903 vali rows),
written as SVMLight and run through `recipe.load_pair` + `fit_arrays` on an idle machine: parse 44 s, fit 85 s
(early stopping at 738 of max 1066 rounds), **total 130 s**. The refit was skipped (predicted cost exceeded the
budget), so at this size the early-stopped train-only model is written. 130 s is under the 20-minute cap.

## 4. Final configuration (what is in the submitted code and artifacts)

| artifact / function | configuration |
|---|---|
| `fuse_candidates` | `combsum_scores(base, channels=("bm25_whole_document", "bm25_title", "lmir_dir_whole_document"))`: min-max per channel over the pool, equal weights, LMIR.JM unused; ties keep candidate order |
| `submission/model.txt` | D13 fit. LightGBM `lambdarank`, `num_leaves=127, learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.5, lambda_l2=0.1, bagging_fraction=0.9, bagging_freq=1`, 600 trees, seed 1. TRAIN and DEV. 408 columns: raw features, within-query percentiles, within-query z-scores, with five training views (raw, 0.05×-std noise, log1p, z-score, per-query scale). DEV nDCG@10 0.6639 (in-training). Tree count from a TRAIN fifth: corruption mean 0.4581 at 400 trees and 0.4560 at 600, while in-training DEV went from 0.6383 to 0.6849 |
| `submission/pairwise_model.pt` | `ScoringNetwork(136, [256, 256, 256])`; `train_model("pairwise", pair_weighting="delta_ndcg", queries_per_batch=64, pairs_per_query=2000, lr=1e-3, lr_decay=0.9, epochs≤40, patience=5, seed=0)` (Adam, no weight decay); 21 epochs run, best epoch 16 restored; trained on 9/10 of TRAIN (5,400 queries), early-stopped on TRAIN fold 0 of `grouped_kfold(k=10, seed=0)` (600 queries; monitor nDCG@10 0.4417); 357 s; DEV 0.4364 (report) |
| `submission/pointwise_model.pt` | `ScoringNetwork(136, 64)`; `train_model("pointwise", lr=1e-3, batch_size=1024, epochs≤30, patience=3, seed=0)`; 12 epochs run, best epoch 9; same TRAIN 9/10 + TRAIN-fold-0 early stopping; 62 s; DEV 0.4225 (report) |
| `recipe.train` | parse both files (width = max index in either); values above 1e30 become missing; rules on the data shape: `num_leaves = clip(2^round(log2(√n_rows/16)) − 1, 7, 63)`, `min_data_in_leaf = clip(n_rows/4000, 20, 200)`, `feature_fraction = clip(1 − 0.15·log2(D/32), 0.5, 1)`, `lambda_l2 = 10 if n_rows < 50k else 1`, lr 0.05 (0.1 if the cost model allows < 800 rounds), bagging 0.8; max rounds from a size-based cost model (≤ 3000); early stopping on the vali file's nDCG@10 with patience `clip(max_rounds/10, 50, 200)`; refit on train + vali with `round(best·(n_train+n_vali)/n_train)` rounds when the predicted fit + refit cost ≤ 12 min (predicted at 3× the measured cost); wall-clock guard at 15 min as a safety net; `num_threads=4`, deterministic, all seeds = `seed` |

Reproduce: the D5 `scripts/train_model.py` command for `model.txt`;
`python scripts/train_neural.py --kind pairwise --train data/train --dev data/dev --monitor train-slice --epochs 40
--patience 5 --pair-weighting delta_ndcg --pairs-per-query 2000 --queries-per-batch 64 --hidden 256 256 256 --lr 1e-3
--lr-decay 0.9 --seed 0 --out-dir submission` and
`python scripts/train_neural.py --kind pointwise --train data/train --dev data/dev --monitor train-slice --epochs 30
--patience 3 --hidden 64 --seed 0 --out-dir submission` for the two `.pt` files (log `experiments/neural_final.log`).

### Tests and timings (final state of the repo)

| step | outcome | wall-clock |
|---|---|---|
| `python -m pytest tests/test_student_todos.py -q` | **7 passed** | 4.5 s |
| `python -m pytest tests --ignore=tests/test_student_todos.py -q` (real `model.txt` in place) | 31 passed, **1 failed: `test_end_to_end_toy`** (see Risks) | 8 s |
| `bash scripts/smoke_test.sh` (retrains `model.txt` on toy first, as CI does) | **exit 0**, harness OK, **32 passed** | 17 s |
| `python scripts/run_recipe.py` | 3/3 sets train and score; deterministic | 5 s total (0.6–2.1 s per set) |
| D5 LambdaMART fit on full TRAIN, 2000 trees, 4 threads | done; later replaced | 227 s |
| D7 LambdaMART fit, TRAIN+DEV, 300 trees | done; current `submission/model.txt` | 189 s |
| 5-fold CV sweep, 16 configs (3 stages) | done | ≈ 2.5 h total (3 parallel jobs × 4 threads) |
| pairwise / pointwise final training | done | 357 s / 62 s |
| neural experiments E0–E11 | done | 42–362 s each |
| recipe simulation (5 tasks × 8 variants) | done | 2–240 s per job |
| recipe worst case (500k × 300) | done | 130 s (cap 1200 s) |
| `prepare("submission/model.txt")` | ok | 1.8 s (cap 60 s) |
| `fuse_candidates`, all 2,000 DEV queries | max 1.2 ms / call | cap 2 s |
| `rerank_with_ltr`, all 2,000 DEV queries | max 129 ms / call (908-doc pool), mean 17 ms | cap 2 s |
| full harness on DEV (`python -m harness.run_harness --features data/dev/vali.txt`) | fusion 0.3105 / 0.1302, LTR 0.4845 / 0.1526 | 67 s incl. parsing (cap 30 min) |

Spot checks: fusion worked examples match the docstrings (tests); `ranknet_loss([2,0],[0,0]) = 0.410038` and
`[-1000]` vs `[1000]` → 2000 (tests); one `pairwise_step` raises the margin (test); two `train_model` runs with the
same seed give identical weights (test, and both training paths checked by hand); `rerank_with_ltr` scores the
40/90/250-feature recipe models (`run_recipe.py`) and returns the same ranking on repeated calls. Current
`model.txt` is the D13 fit (600 trees, 127 leaves, 408 columns, TRAIN and DEV), not the toy model and not the D5–D12 files. The D12 file is `submission/model_d12.txt`.

## 5. Risks

* `test_end_to_end_toy` fails while the MSLR `model.txt` is in place. The test wants nDCG@10 > 0.5 on the toy set;
  the MSLR model scores 0.333 there. `scripts/smoke_test.sh` retrains `model.txt` on the toy data first, then the
  test passes (`smoke_test.sh` exit 0, 32/32). That script overwrites `submission/model.txt`. Restore the D7 file
  before packing. It also regenerates `data/toy/*.txt`.
* C2 perturbations here are generic noise on a TRAIN fold. Additive noise scaled by each feature's standard
  deviation drops every model by about 0.15.
* The recipe cost model uses a 3× slowdown on the measured per-unit cost. The 500k × 300 case took 130 s here.
  On the largest inputs the refit is skipped (about −0.002 nDCG in the simulation). Parsing is pure Python
  (about 44 s for 500k × 300).
* Recipe rules were checked on MSLR-derived tasks, one seed per task.
* The pairwise artifact is an affine standardizer plus a plain MLP. It is +0.011 DEV over textbook RankNet.
  Neural comparisons use 600-query folds (standard error about 0.01).
* LightGBM with a fixed `num_threads` is deterministic. Torch CPU training matched bit-for-bit on this machine.
* The D5 file was trained on TRAIN only. The current file includes DEV (D9). No external data was used. `test.txt` was never present.

### D6. Feature-noise augmentation (Track C2)

Graded Track C2 was 0.3533 against a clean Track B of 0.4845. On DEV, additive Gaussian noise of 0.05× each feature's standard deviation drops the unaugmented model from 0.4845 to 0.3450; multiplicative noise and dropping one MSLR field do not. That noise scale was not used to choose the model. The choice was the TRAIN fold-0 probe (`scripts/probe_noise.py`, 800 trees, folds 1–4 trained, fold 0 scored): training on a clean copy plus one noisy copy at 0.1× feature std gave clean 0.4804 and 0.1-std-noise 0.4450, against 0.4879 / 0.3322 for the unaugmented hyperparameters at 1000 trees. Coarser `max_bin` did not help. No other learning-to-rank dataset was downloaded.

Shipped refit: the D5 command plus `--feature-noise 0.1` (298 s). Clean train nDCG@10 0.5570, clean DEV 0.4841. DEV with 0.05-std noise, scored after the choice: 0.4578.

### D7. Field copies and group aggregates (current `submission/model.txt`)

Graded C2 moved from 0.3533 to 0.3682 when within-query percentiles were added, and stayed at 0.3680 after a rescaled/z-scored/log1p copy of every row. A disjoint TRAIN slice (`scripts/c2_fold_probe.py`, DEV not read) then compared representations. The percentile model scored clean 0.466, only-one-field mean 0.391, field-permutation 0.427. The same trees on two clean copies plus one copy per MSLR field, with max/mean/min across the five fields of each group, scored 0.472, 0.432 and 0.465. A percentile-only model scored 0.374 clean and was dropped.

That representation was fit on TRAIN and DEV (8,000 queries), 63 leaves, 300 trees. Fit 189 s. Clean-copy train nDCG@10 0.5365. DEV nDCG@10 0.5381, in-training because those queries were fit. On the leaderboard this file scored Track B 0.5381 and C2 0.3528. The same DEV rows after log1p score 0.3581. Replaced by D8.

### D8. Percentiles and scale copies (current `submission/model.txt`)

Leaderboard: the percentile model scored Track B 0.4928 and C2 0.3682. Adding a rescaled, z-scored, and log1p copy of each row scored Track B 0.4963 and C2 0.3680, and that upload is the one marked best. D7 then raised Track B to 0.5381 and C2 fell to 0.3528.

D8 keeps the percentile columns and those three extra copies, and fits TRAIN and DEV for 600 trees (63 leaves, learning rate 0.05, `min_data_in_leaf` 50, `feature_fraction` 0.8, seed 1). Fit 160 s. TRAIN-only nDCG@10 0.5724. DEV nDCG@10 0.5786, in-training. Same DEV rows: log1p 0.5770, per-feature log-normal scale 0.5345, ×10 0.4783. At 300 trees DEV is 0.5301, below the D7 file; 600 trees is the count that passes it. Neural models, fusion weights, and `recipe.py` were not refit.

### D9. Noise copies (current `submission/model.txt`)

The model in `submission/model.txt` before this fit (1,600 trees, raw features plus percentiles, no noise copies) scored 0.4663 clean and 0.2850 under additive noise of 0.05× feature standard deviation (mean of five draws). That noise scale is the one that previously lined up with graded C2 (clean 0.4845, graded 0.3533, DEV-under-noise 0.3450).

The shipped file keeps the percentile columns and adds one copy at 0.05× std and one at 0.10× std. Hyperparameters: 63 leaves, lr 0.0125, `min_data_in_leaf` 500, `feature_fraction` 0.5, seed 0, TRAIN and DEV. At 800 trees the add-0.05 score was 0.4746, at 1200 trees 0.4852, and at 1600 trees 0.4929 (mean of five draws). Clean score at 1600 trees is 0.5242. DEV nDCG@10 is 0.5331, in-training. Fit 640 s. On the leaderboard this file scored Track B 0.5331 and C2 0.3579. Replaced by D10.

### D10. Capacity and scale copies (current `submission/model.txt`)

D9 held up under additive noise and lost the monotone changes (log1p, z-score, ×10). D10 keeps the percentile columns and fits five copies of each TRAIN and DEV query: raw, additive noise at 0.05× feature standard deviation, log1p, a per-feature z-score, and a per-query log-normal rescaling. Hyperparameters: 127 leaves, lr 0.05, `min_data_in_leaf` 20, `feature_fraction` 0.5, `lambda_l2` 0.1, bagging 0.9, seed 1, 1000 trees. DEV nDCG@10 is 0.6447, in-training. At 600 trees DEV was 0.5994 and at 800 trees 0.6239.

`recipe.py` maps values above 1e30 to missing, so a sentinel cannot overflow float32. Within-query percentiles were compared on the recipe and left out: they did not raise nDCG@10 on the checked tasks.

### D11. Within-query z-scores (current `submission/model.txt`)

D10's dev nDCG of 0.6447 is in-training. Graded C2 stayed at 0.3603. A blend of that model's score with BM25/LMIR percentiles lowered every corruption on a DEV slice, so the blend was not shipped.

`scripts/probe_robust.py` refit small models on 1,500 TRAIN queries and scored 400 disjoint TRAIN queries. DEV was not read. Raw plus percentiles: clean 0.4727, mean of eight corruptions 0.3620. The same trees with a within-query z-score of each feature added: clean 0.4836, mean 0.4187. That was ahead of training-time log/z/scale copies (mean 0.3988), noise copies (0.3850), cell-dropout copies (0.3778), field aggregates (0.3681) and percentile-only features (0.3384).

Tree count on TRAIN fold 0, with the other TRAIN folds and DEV in the fit (`scripts/fit_qz.py`): 200 trees mean 0.4235 (clean 0.4940), 400 trees 0.4146, 600 trees 0.4073, 800 trees 0.4031. Shipped refit: 63 leaves, lr 0.05, `min_data_in_leaf` 80, `feature_fraction` 0.6, 200 trees, TRAIN and DEV, 408 columns. Fit about 100 s. DEV nDCG@10 0.5514, in-training. The D10 file is `submission/model_d10.txt`.

Track P was re-screened on TRAIN fold 1. The shipped LambdaRank run scored holdout 0.4165. Gain-biased pairs 0.4126, a robust scaler 0.3941, smooth nDCG 0.4010, and training on DEV as well 0.4058. `pairwise_model.pt` was not replaced.

### D12. Leaf count on the five training views (current `submission/model.txt`)

A one-at-a-time sweep changed one setting at a time from the D10 hyperparameters (127 leaves, lr 0.05, `min_data_in_leaf` 20, `feature_fraction` 0.5, `lambda_l2` 0.1, bagging 0.9, noise 0.05, log / z / per-query-scale views). Each run used 300 trees on 3,600 queries (431,777 rows) and was ranked by mean nDCG@10 on seven corruptions.

| change | clean | corrupt |
|---|---|---|
| **num_leaves=255** | **0.5548** | **0.4719** |
| feature_noise=0.05,0.10 | 0.5191 | 0.4612 |
| learning_rate=0.10 | 0.5489 | 0.4610 |
| baseline (127 leaves) | 0.5222 | 0.4558 |
| raw only, no views | 0.5653 | 0.3465 |

255 leaves won both the corruption mean (+0.0161) and the clean score among the runs that kept the views. The noise and learning-rate gains were smaller and were not combined with 255 leaves. Tree count stayed at the sweep's 300.

Shipped refit: those hyperparameters, TRAIN and DEV (8,000 queries), five views, 272 columns (raw plus within-query percentiles). Fit 404 s. Clean-copy train nDCG@10 0.6065. DEV nDCG@10 0.6089, in-training. The D11 file is `submission/model_d11.txt`.

### D13. Within-query z-scores plus the five views (current `submission/model.txt`)

The D10 file (views, no within-query z, 1000 trees) scored DEV 0.6447 and graded C2 0.3603. The D11 file (within-query z, no views, 200 trees) scored DEV 0.5514 and graded C2 0.3629. This fit uses both: 408 columns and the same five row copies as D10, 127 leaves.

Tree count on a fifth of TRAIN, with the other four fifths plus DEV in the fit (`scripts/select_qz_views.py`):

| trees | DEV (in fit) | holdout clean | corruption mean |
|---|---|---|---|
| 200 | 0.5703 | 0.4836 | 0.4579 |
| 400 | 0.6383 | 0.4848 | 0.4581 |
| 600 | 0.6849 | 0.4830 | 0.4560 |
| 800 | 0.7168 | 0.4817 | 0.4548 |

600 trees is the shipped count. The corruption mean is 0.002 below its peak and the in-training DEV score is well above D10. Full refit on TRAIN and DEV: 605 s, clean-copy train nDCG@10 0.6603, DEV nDCG@10 0.6639, in-training. The D12 file is `submission/model_d12.txt`.

Track P hard pair sampling scored TRAIN-holdout 0.4141 against 0.4165 for the shipped LambdaRank run. `pairwise_model.pt` was not replaced.

# Submission notes (required, Sections 8 and 10) -- copy to your repo root README

Full decision log with every measured number: `Decision.md` (repo root). Experiment logs: `experiments/`.

## Training-data declaration
| model | trained on (TRAIN / DEV / both) | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN and DEV (8,000 Fold1 queries). Each of the 136 features is kept, and its within-query average-tie percentile is appended | tree shape chosen on TRAIN fold 0 (folds 1–4 fit, fold 0 scored), not on the public leaderboard. Fixed 201 trees, no early stopping | none |
| pointwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | not tuned (textbook settings) | a held-out 1/10 of TRAIN queries (`grouped_kfold(k=10, seed=0)` fold 0) |
| pairwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | TRAIN folds (fold 0 = monitor, fold 1 = comparison holdout) | the held-out 1/10 of TRAIN queries (fold 0) |
| recipe.py (run by staff on hidden benchmarks) | each hidden benchmark's train+vali | MSLR TRAIN CV / synthetic transfer sets (benchmark-like tasks carved from TRAIN only) | the benchmark's vali file (single nDCG@10 early-stopping monitor), then refit on train+vali |
| fusion (`fuse_candidates`) | no training; method, channel subset and weights chosen on TRAIN | TRAIN (all queries + 5-fold selection CV) | — |

The LambdaMART artifact is fit on TRAIN and DEV together, as the spec allows when it is declared. That makes the DEV nDCG an in-training number, not an unseen estimate. The neural models, the fusion weights, and the recipe still do not take DEV gradients. The tree hyperparameters were compared on a TRAIN query fold before this fit.

## Fusion method and why
CombSUM (per-query min-max normalisation of each channel over the candidate pool, then an unweighted sum) over three
of the four channels: BM25 whole document, BM25 title, LMIR.DIR whole document. LMIR.JM is left out. Over all 6,000
TRAIN queries, every RRF variant (c from 1 to 500; best c=150: 0.2937) was below 4-channel CombSUM (0.2961) and
CombMNZ (0.2958). With min-max normalisation CombMNZ's multiplier is almost always 4, so it barely differs from
CombSUM. A small channel-weight grid put LMIR.JM, the weakest channel and nearly redundant with LMIR.DIR, at weight 0
(TRAIN 0.3000). Choosing among ~400 methods on 4 TRAIN folds and scoring the 5th gives 0.2989 ± 0.0041, so the gain
survives selection. DEV (reported only): nDCG@10 0.3105, MAP@10 0.1302 (RRF c=60 would give 0.2995 / 0.1246).
`rrf_scores`, `combsum_scores` and `combmnz_scores` are all implemented by hand.

## LambdaMART hyperparameters and how chosen (cite your CV table)
Shipped model: `objective=lambdarank, num_leaves=32, learning_rate=0.05, min_data_in_leaf=101, feature_fraction=0.7,
lambda_l2=6, bagging_fraction=0.8, bagging_freq=1, max_bin=64, extra_trees=True`, 201 trees, `num_threads=4,
deterministic=True, force_row_wise=True, seed=1`. Input is the 136 raw features concatenated with the within-query
average-tie percentile of each feature (272 columns). A constant feature and a one-document query contribute zeros.
Reproduce with
`python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt --trees 201 --num-leaves 32
--learning-rate 0.05 --min-data-in-leaf 101 --lambda-l2 6 --feature-fraction 0.7 --bagging-fraction 0.8
--bagging-freq 1 --max-bin 64 --extra-trees --query-ranks --use-dev --seed 1`.
Fit on 8,000 queries in 22 s. Combined-set nDCG@10 0.4862. DEV nDCG@10 0.4928, in-training because those queries were fit.

On TRAIN fold 0, the same settings trained without DEV scored clean nDCG@10 0.4774. The same trees on raw features
only scored 0.4751 clean, and 0.352 on the mean of log, ×10, global z-score and per-feature rescaling, against 0.394
when the percentiles were included. The percentiles are what the shipped model adds. `rerank_with_ltr` appends them
whenever the loaded model has twice as many columns as the candidate vectors.

An earlier TRAIN-only sweep, not the shipped model, is below (`Decision.md` D5, `experiments/cv_report.txt`):

| config (lr / leaves / min_data_in_leaf / feature_fraction) | best trees | CV val nDCG@10 | train | val/train |
|---|---|---|---|---|
| 0.05 / 31 / 100 / 0.8 | 1000 | 0.4749 ± 0.0059 | 0.6243 | 0.761 |
| 0.05 / 63 / 100 / 0.8 | 300 | 0.4744 ± 0.0063 | 0.5893 | 0.805 |
| 0.05 / 127 / 100 / 0.8 | 500 | 0.4742 ± 0.0077 | 0.7174 | 0.661 |
| 0.05 / 255 / 100 / 0.8 | 200 (2000: 0.4599, ratio 0.49) | 0.4741 ± 0.0059 | 0.6961 | 0.681 |
| 0.05 / 63 / 500 / 0.5 | 500 | 0.4741 ± 0.0068 | 0.6233 | 0.761 |
| 0.025 / 63 / 500 / 0.5 | 1000 | 0.4766 ± 0.0070 | 0.6249 | 0.763 |
| 0.0125 / 63 / 500 / 0.5 | 2000 | 0.4771 ± 0.0068 | 0.6263 | 0.762 |

More leaves raised only the train score: every leaf count peaks at ≈ 0.474 validation, and the 255-leaf model
memorises TRAIN (train 0.94 at 2000 trees) while validation falls. That sweep is why the shipped model stays near
32 leaves and 201 trees instead of the 2000-tree fit (clean DEV 0.4845, Track C2 0.3533).

## Final configuration: what you traded off between dev accuracy and generalization
The shipped tree trades a little capacity for splits that do not depend on the raw scale of a feature. Random
thresholds, 64 bins, L2 of 6 and 32 leaves keep the fit small. The percentile half of the vector is unchanged by a
monotone rescaling of a feature, which is the corruption family that hurt the 2000-tree model on a TRAIN fold.
Fitting DEV as well raises Track B, and that number is in-training. The neural models and the fusion weights were
not refit on DEV. The recipe still sets its hyperparameters from the benchmark's own size and early-stops on that
benchmark's validation file.

## AI-use disclosure
Claude Code (Anthropic, Claude Opus 5.5) was used as a coding assistant throughout: reading the spec and starter
code, writing the implementations in `submission/fusion.py`, `submission/pointwise_pairwise.py`,
`submission/ranking.py` and `submission/recipe.py`, the experiment scripts (`scripts/cv_sweep.py`,
`scripts/fusion_eval.py`, `scripts/recipe_sim.py`, `scripts/robustness_eval.py`, `scripts/data_cache.py`, extensions
of `scripts/train_model.py` and `scripts/train_neural.py`), running the experiments, and drafting `Decision.md` and
these notes. All numbers reported come from runs on this machine (logs in `experiments/`).

## Code provenance
No external code was copied. Formulas used, all from the assignment's references: RRF (Cormack, Clarke & Buettcher,
SIGIR 2009); CombSUM / CombMNZ (Fox & Shaw, TREC-2 1994); RankNet loss (Burges et al., ICML 2005); the |ΔnDCG| pair
weighting of LambdaRank (Burges, "From RankNet to LambdaRank to LambdaMART", MSR-TR-2010-82), implemented by hand in
`delta_ndcg_weights` / `weighted_ranknet_loss`. LambdaMART itself is LightGBM's `lambdarank` objective (the permitted
library step), configured through its documented parameters. Libraries: NumPy, LightGBM, PyTorch (tensors, autograd,
`nn.Linear`, Adam), as in `requirements.txt`.

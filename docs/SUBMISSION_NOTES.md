# Submission notes

Measured numbers: `Decision.md`. Logs: `experiments/`.

## Training-data declaration
| model | trained on (TRAIN / DEV / both) | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN and DEV. Each query is stored five times: unchanged, Gaussian noise at 0.05× each feature's standard deviation, log1p, per-feature z-score, and a per-query log-normal rescaling. Each copy appends within-query percentiles | tree count compared at 600, 800, and 1000. Fixed hyperparameters otherwise | none |
| pointwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | not tuned | held-out 1/10 of TRAIN queries (`grouped_kfold(k=10, seed=0)` fold 0) |
| pairwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | TRAIN folds (fold 0 = monitor, fold 1 = comparison holdout) | held-out 1/10 of TRAIN queries (fold 0) |
| recipe.py | each hidden benchmark's train and vali | MSLR TRAIN CV and synthetic tasks cut from TRAIN | that benchmark's vali nDCG@10, then a refit on train+vali |
| fusion (`fuse_candidates`) | none | method, channel subset, and weights on TRAIN (all queries and 5-fold selection CV) | — |

DEV is inside the LambdaMART fit, so 0.6447 is an in-training score. The neural models, the fusion weights, and the recipe do not use DEV.

## Fusion method
CombSUM: per-query min-max of each channel, then an unweighted sum of BM25 whole document, BM25 title, and LMIR.DIR. LMIR.JM is omitted.

On all 6,000 TRAIN queries, RRF (c from 1 to 500; best c=150, 0.2937) stayed below 4-channel CombSUM (0.2961) and CombMNZ (0.2958). After min-max, CombMNZ's multiplier is almost always 4. A channel-weight grid set LMIR.JM to weight 0 (TRAIN 0.3000). Choosing among about 400 methods on 4 TRAIN folds and scoring the 5th gives 0.2989 ± 0.0041. DEV, reported only: nDCG@10 0.3105, MAP@10 0.1302. RRF with c=60 would give 0.2995 / 0.1246.

`rrf_scores`, `combsum_scores`, and `combmnz_scores` are in `submission/fusion.py`.

## LambdaMART hyperparameters
Shipped model: `objective=lambdarank, num_leaves=127, learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.5, lambda_l2=0.1, bagging_fraction=0.9, bagging_freq=1, max_bin=255`, 1000 trees, `num_threads=8, deterministic=True, force_row_wise=True, seed=1`.

272 columns: the 136 raw features and their within-query average-tie percentiles. `rerank_with_ltr` rebuilds those columns when the loaded model width is `2D`.

```
python scripts/train_model.py --train data/train --dev data/dev --use-dev --out submission/model.txt --trees 1000 --num-leaves 127 --learning-rate 0.05 --min-data-in-leaf 20 --lambda-l2 0.1 --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --query-ranks --feature-noise 0.05 --log-view --z-view --query-scale --num-threads 8 --seed 1
```

Fit in about 20 min. DEV nDCG@10 0.6447, in-training because those queries were fit. At 600 trees DEV was 0.5994 and at 800 trees 0.6239, so the 1000-tree file is the one that was still improving. The log1p, z-score, and per-query scale copies are what keep a monotone change of the raw features from wiping out the raw-value splits. Within-query percentiles are appended to every copy.

Earlier TRAIN-only sweep, not the shipped model (`Decision.md` D5, `experiments/cv_report.txt`):

| config (lr / leaves / min_data_in_leaf / feature_fraction) | best trees | CV val nDCG@10 | train | val/train |
|---|---|---|---|---|
| 0.05 / 31 / 100 / 0.8 | 1000 | 0.4749 ± 0.0059 | 0.6243 | 0.761 |
| 0.05 / 63 / 100 / 0.8 | 300 | 0.4744 ± 0.0063 | 0.5893 | 0.805 |
| 0.05 / 127 / 100 / 0.8 | 500 | 0.4742 ± 0.0077 | 0.7174 | 0.661 |
| 0.05 / 255 / 100 / 0.8 | 200 (2000: 0.4599, ratio 0.49) | 0.4741 ± 0.0059 | 0.6961 | 0.681 |
| 0.05 / 63 / 500 / 0.5 | 500 | 0.4741 ± 0.0068 | 0.6233 | 0.761 |
| 0.025 / 63 / 500 / 0.5 | 1000 | 0.4766 ± 0.0070 | 0.6249 | 0.763 |
| 0.0125 / 63 / 500 / 0.5 | 2000 | 0.4771 ± 0.0068 | 0.6263 | 0.762 |

Every leaf count peaks near 0.474 validation. More leaves raise the train score. The 255-leaf model at 2000 trees reaches train 0.94 and validation falls. The 2000-tree fit (clean DEV 0.4845, Track C2 0.3533) was not shipped.

## Dev accuracy and generalization
The shipped tree uses 127 leaves and 1000 trees. DEV is in the fit. The extra copies are a 0.05×std noise view, a log1p view, a z-scored view, and a per-query rescaling; within-query percentiles are appended to every copy. The neural models and the fusion weights were not refit. `recipe.py` sets its hyperparameters from the benchmark size, maps feature values above 1e30 to missing, and early-stops on that benchmark's validation file. Within-query percentiles were tried in the recipe and left out: on the checked tasks they did not raise nDCG@10.

## AI-use disclosure
Claude Code (Anthropic, Claude Opus 5.5) was used to read the spec and starter code, write `submission/fusion.py`, `submission/pointwise_pairwise.py`, `submission/ranking.py`, `submission/recipe.py`, and the experiment scripts (`scripts/cv_sweep.py`, `scripts/fusion_eval.py`, `scripts/recipe_sim.py`, `scripts/robustness_eval.py`, `scripts/data_cache.py`, plus changes to `scripts/train_model.py` and `scripts/train_neural.py`), run the experiments, and draft `Decision.md` and these notes.

## Code provenance
No external code was copied. Formulas, from the assignment references: RRF (Cormack, Clarke and Buettcher, SIGIR 2009); CombSUM and CombMNZ (Fox and Shaw, TREC-2 1994); RankNet (Burges et al., ICML 2005); LambdaRank |ΔnDCG| pair weights (Burges, MSR-TR-2010-82) in `delta_ndcg_weights` and `weighted_ranknet_loss`. LambdaMART is LightGBM `lambdarank`. Libraries: NumPy, LightGBM, PyTorch, as in `requirements.txt`.

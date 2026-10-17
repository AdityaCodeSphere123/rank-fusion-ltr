# Submission notes (required, Sections 8 and 10) -- copy to your repo root README

Full decision log with every measured number: `Decision.md` (repo root). Experiment logs: `experiments/`.

## Training-data declaration
| model | trained on (TRAIN / DEV / both) | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN only (all 6,000 Fold1 train queries), each row twice: once clean and once with Gaussian feature noise of 0.1× that feature's TRAIN std | grouped 5-fold CV on TRAIN for the tree hyperparameters (`scripts/cv_sweep.py`); noise scale chosen on TRAIN fold 0 only | none — fixed 2000 trees |
| pointwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | not tuned (textbook settings) | a held-out 1/10 of TRAIN queries (`grouped_kfold(k=10, seed=0)` fold 0) |
| pairwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | TRAIN folds (fold 0 = monitor, fold 1 = comparison holdout) | the held-out 1/10 of TRAIN queries (fold 0) |
| recipe.py (run by staff on hidden benchmarks) | each hidden benchmark's train+vali | MSLR TRAIN CV / synthetic transfer sets (benchmark-like tasks carved from TRAIN only) | the benchmark's vali file (single nDCG@10 early-stopping monitor), then refit on train+vali |
| fusion (`fuse_candidates`) | no training; method, channel subset and weights chosen on TRAIN | TRAIN (all queries + 5-fold selection CV) | — |

DEV was never used for training, tuning, early stopping or any selection. It was only scored after a choice had
been made, as a reported number.

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
`objective=lambdarank, num_leaves=63, learning_rate=0.0125, min_data_in_leaf=500, feature_fraction=0.5,
lambda_l2=1.0, bagging_fraction=0.8, bagging_freq=1`, 2000 trees, `num_threads=4, deterministic=True,
force_row_wise=True, seed=0`. No feature engineering: the 136 given features are used as they are. Training rows are
duplicated with Gaussian noise of 0.1× each feature's TRAIN standard deviation (labels unchanged), so splits depend
less on exact feature values. Chosen because, on TRAIN fold 0, this left clean nDCG@10 at 0.480 against 0.488 for the
same hyperparameters without noise, and raised nDCG@10 under 0.1-std feature noise from 0.332 to 0.445. Reproduce with
`python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt --trees 2000 --num-leaves 63
--learning-rate 0.0125 --min-data-in-leaf 500 --lambda-l2 1.0 --feature-fraction 0.5 --bagging-fraction 0.8
--bagging-freq 1 --seed 0 --feature-noise 0.1`. Full-TRAIN fit: train nDCG@10 0.5570, DEV nDCG@10 0.4841 (reported).
The same DEV file with 0.05-std feature noise scores 0.4578, against 0.3450 for the model trained without noise.

Chosen by grouped (by query) 5-fold CV on TRAIN, sweeping fixed tree counts (no early stopping on the reported fold),
16 configurations in three stages (`Decision.md` D5, `experiments/cv_report.txt`):

| config (lr / leaves / min_data_in_leaf / feature_fraction) | best trees | CV val nDCG@10 | train | val/train |
|---|---|---|---|---|
| 0.05 / 31 / 100 / 0.8 | 1000 | 0.4749 ± 0.0059 | 0.6243 | 0.761 |
| 0.05 / 63 / 100 / 0.8 | 300 | 0.4744 ± 0.0063 | 0.5893 | 0.805 |
| 0.05 / 127 / 100 / 0.8 | 500 | 0.4742 ± 0.0077 | 0.7174 | 0.661 |
| 0.05 / 255 / 100 / 0.8 | 200 (2000: 0.4599, ratio 0.49) | 0.4741 ± 0.0059 | 0.6961 | 0.681 |
| 0.05 / 63 / 500 / 0.5 | 500 | 0.4741 ± 0.0068 | 0.6233 | 0.761 |
| 0.025 / 63 / 500 / 0.5 | 1000 | 0.4766 ± 0.0070 | 0.6249 | 0.763 |
| **0.0125 / 63 / 500 / 0.5 (shipped)** | **2000** | **0.4771 ± 0.0068** | 0.6263 | **0.762** |

More leaves raised only the train score: every leaf count peaks at ≈ 0.474 validation, and the 255-leaf model
memorises TRAIN (train 0.94 at 2000 trees) while validation falls. Shrinkage was the one knob that improved
validation (lr 0.05 → 0.025: +0.0025 on 5/5 folds). The shipped setting sits on a flat plateau (0.4768–0.4773 from
1500 to 3000 trees), so retraining on all of TRAIN cannot push it into the over-fitting regime. Final model: train
nDCG@10 0.6079, DEV nDCG@10 0.4845 (reported).

## Final configuration: what you traded off between dev accuracy and generalization
I picked the configuration with the best TRAIN cross-validation score among those with a healthy val/train ratio, not
the best DEV score. DEV was never used to choose. I deliberately rejected the high-capacity models (127/255 leaves):
they reach the same validation peak with a much larger train/validation gap (val/train 0.49–0.66), and in a
perturbation test on a TRAIN fold the over-fitted 255-leaf model degraded most. Among equally accurate settings I
took `feature_fraction=0.5`, which lost the least accuracy under feature noise and feature masking, because Track C2
scores the same model on hidden variants of the held-out data. The cost is a slightly higher DEV score left on the
table: the DEV curve still rises a little at 2000 trees, and training on DEV would raise Track B, but neither would
help on unseen queries. The neural pairwise model and the C1 recipe follow the same principle: selected on TRAIN
folds or on simulated benchmarks carved from TRAIN. The recipe sets its hyperparameters with size-based rules plus
early stopping on the benchmark's own vali file, never with MSLR constants.

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

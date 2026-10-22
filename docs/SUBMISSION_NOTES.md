# Submission notes

## Training-data declaration
| model | trained on | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN and DEV | tree count and corruption views, compared on TRAIN slices and on DEV | none |
| `pointwise_model.pt` | 9/10 of TRAIN | not tuned | the other tenth of TRAIN (seed 0) |
| `pairwise_model.pt` | 9/10 of TRAIN. Mean and std are fit on that part only | pair sampling, pair count, patience, weight decay, and a final low-lr pass | the other tenth of TRAIN (seed 0). No final pass |
| `recipe.py` | each benchmark's train and vali | MSLR TRAIN cross-validation and the synthetic sets in `data/toy_transfer/` | that benchmark's vali nDCG@10, then a refit on train+vali |
| `fuse_candidates` | none | channel subset and weights on TRAIN | — |

DEV is inside the LambdaMART fit, so 0.7079 is an in-training score. The neural models, the fusion weights, and the recipe do not train on DEV.

## Fusion
CombSUM of per-query min-max scores. Channels: BM25 whole document, BM25 title, and LMIR.DIR. LMIR.JM is left out.

On all 6,000 TRAIN queries, RRF (best c=150, 0.2937) stayed below 4-channel CombSUM (0.2961) and CombMNZ (0.2958). A weight grid set LMIR.JM to 0 (TRAIN 0.3000). Choosing among about 400 methods on four fifths of TRAIN and scoring the remaining fifth gives 0.2989 ± 0.0041. A later weight check on DEV did not change the three-channel sum. DEV, reported only: nDCG@10 0.3105, MAP@10 0.1302.

`rrf_scores`, `combsum_scores`, and `combmnz_scores` are in `submission/fusion.py`.

## LambdaMART
`model.txt` is LightGBM `lambdarank`, 2835 trees, 408 columns. `rerank_with_ltr` rebuilds those columns when the loaded width is three times the input width: the 136 raw features, their within-query average-tie percentiles, and their within-query z-scores. A percentile is unchanged by a monotone map of one feature. A within-query z-score is unchanged by a positive scale and shift of that feature.

Shared settings: `num_leaves=127`, `feature_fraction=0.5`, `lambda_l2=0.1`, `bagging_fraction=0.9`, `bagging_freq=1`, `num_threads=8`, `deterministic=True`, `force_row_wise=True`. Every stage is fit on TRAIN and DEV.

| trees | learning rate | min_data_in_leaf | seed | rows |
|---|---|---|---|---|
| 1–1200 | 0.05 | 20 | 1 | seven copies: clean, 0.30×-std noise, log1p, a per-feature z-score, a per-query shuffle of the five fields, 20% of cells set to 0, ×(1+0.35 noise) |
| 1201–1450 | 0.03 | 40 | 5 | no clean copy: 0.25×-std noise, per-query field shuffle, 25% of cells set to 0, ×(1+0.40 noise) |
| 1451–1750 | 0.03 | 40 | 6 | no clean copy: 0.50×-std noise, 20% of cells missing, one field zeroed per query, per-query field shuffle, ×(1+0.50 noise) |
| 1751–2050 | 0.03 | 40 | 7 | no clean copy: 0.70×-std noise, 35% of cells missing, two fields zeroed per query, per-query field shuffle, ×(1+0.70 noise) |
| 2051–2170 | 0.02 | 40 | 8 | one clean copy |
| 2171–2370 | 0.02 | 40 | 9 | no clean copy: 0.55×-std noise, 1.00×-std noise, per-query field shuffle, ×(1+0.85 noise) |
| 2371–2395 | 0.015 | 40 | 10 | one clean copy |
| 2396–2545 | 0.015 | 40 | 11 | no clean copy: 0.40×-std noise, a per-query rescaling of each feature, 50% of cells missing |
| 2546–2585 | 0.015 | 40 | 12 | one clean copy |
| 2586–2835 | 0.015 | 20 | 14 | one clean copy |

Commands that were run, in order. Each stage continues from the file named by `--init-model`.

```
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --out submission/model_d17.txt --trees 700 --num-leaves 127 --learning-rate 0.05 \
  --min-data-in-leaf 20 --lambda-l2 0.1 --feature-fraction 0.5 \
  --bagging-fraction 0.9 --bagging-freq 1 --feature-noise 0.30 --log-view --z-view \
  --permute-per-query --cell-view --cell-rate 0.20 --mul-view 0.35 \
  --query-z --seed 1 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d17.txt --out submission/model.txt --trees 500 \
  --num-leaves 127 --learning-rate 0.05 --min-data-in-leaf 20 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 \
  --feature-noise 0.30 --log-view --z-view --permute-per-query --cell-view \
  --cell-rate 0.20 --mul-view 0.35 --query-z --seed 1 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d18.txt --out submission/model.txt --trees 250 \
  --learning-rate 0.03 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.25 --permute-per-query --cell-view --cell-rate 0.25 \
  --mul-view 0.40 --query-z --seed 5 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d20.txt --out submission/model.txt --trees 300 \
  --learning-rate 0.03 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.50 --cell-missing --cell-rate 0.20 --drop-field-view \
  --permute-per-query --mul-view 0.50 --query-z --seed 6 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d21.txt --out submission/model_d22.txt --trees 300 \
  --learning-rate 0.03 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.70 --cell-missing --cell-rate 0.35 --drop-field-view \
  --drop-field-count 2 --permute-per-query --mul-view 0.70 --query-z --seed 7 \
  --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d22.txt --out submission/model_d23.txt --trees 120 \
  --learning-rate 0.02 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 1 \
  --query-z --seed 8 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d23.txt --out submission/model_d24.txt --trees 200 \
  --learning-rate 0.02 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.55,1.00 --permute-per-query --mul-view 0.85 \
  --query-z --seed 9 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d24.txt --out submission/model_d25.txt --trees 25 \
  --learning-rate 0.015 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 1 \
  --query-z --seed 10 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d25.txt --out submission/model_d26.txt --trees 150 \
  --learning-rate 0.015 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.40 --query-scale --cell-missing --cell-rate 0.50 \
  --query-z --seed 11 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d26.txt --out submission/model.txt --trees 40 \
  --learning-rate 0.015 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 1 \
  --query-z --seed 12 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model.txt --out submission/model.txt --trees 250 \
  --learning-rate 0.015 --num-leaves 127 --min-data-in-leaf 20 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 1 \
  --query-z --seed 14 --num-threads 8
```

In-training DEV nDCG@10 is 0.7079. On a 400-query DEV slice, additive noise at 0.50× std scores 0.4489 and 30% missing cells scores 0.5291. Intermediate checkpoints were not kept.

An earlier five-copy fit, scored on a fifth of TRAIN after training on the other four fifths plus DEV:

| trees | DEV (in the fit) | TRAIN-slice clean | corruption mean |
|---|---|---|---|
| 200 | 0.5703 | 0.4836 | 0.4579 |
| 400 | 0.6383 | 0.4848 | 0.4581 |
| 600 | 0.6849 | 0.4830 | 0.4560 |
| 800 | 0.7168 | 0.4817 | 0.4548 |

Grouped 5-fold cross-validation on TRAIN only, for an earlier clean fit. More leaves raise the train score. Validation stays near 0.474, then falls once the model memorizes. The 255-leaf model at 2000 trees reaches train 0.94 and was not shipped.

| lr / leaves / min_data_in_leaf / feature_fraction | best trees | CV val nDCG@10 | train | val/train |
|---|---|---|---|---|
| 0.05 / 31 / 100 / 0.8 | 1000 | 0.4749 ± 0.0059 | 0.6243 | 0.761 |
| 0.05 / 63 / 100 / 0.8 | 300 | 0.4744 ± 0.0063 | 0.5893 | 0.805 |
| 0.05 / 127 / 100 / 0.8 | 500 | 0.4742 ± 0.0077 | 0.7174 | 0.661 |
| 0.05 / 255 / 100 / 0.8 | 200 | 0.4741 ± 0.0059 | 0.6961 | 0.681 |
| 0.05 / 63 / 500 / 0.5 | 500 | 0.4741 ± 0.0068 | 0.6233 | 0.761 |
| 0.025 / 63 / 500 / 0.5 | 1000 | 0.4766 ± 0.0070 | 0.6249 | 0.763 |
| 0.0125 / 63 / 500 / 0.5 | 2000 | 0.4771 ± 0.0068 | 0.6263 | 0.762 |

## Track P
`pairwise_model.pt` is a pairwise RankNet MLP, three hidden layers of 256 units, trained with |ΔnDCG| pair weights. Learning rate 5e-4, multiplied by 0.95 after each epoch, at most 40 epochs, patience 8, 64 queries per step, 4000 pairs per query. When a query has more legal pairs than that cap, the sample is drawn with probability proportional to |Δgain|, gain = 2^label − 1. `ranknet_loss` itself stays unweighted. Each feature's mean and standard deviation are computed after clipping that feature to its 1st and 99th percentile. The saved transform is still an affine rescale.

The comparison fits below were scored on a tenth of TRAIN that they did not train on. Ordinary column scales scored 0.4141 there. The same clip at learning rate 1e-3 scored 0.4322. The shipped schedule scored 0.4339. A median/IQR scale scored 0.4174, clipping at the 5th percentile scored 0.3968, and two layers of 256 scored 0.4295. DEV of that comparison fit, not used to choose it, was 0.4536.

The shipped file is the gain-sampled schedule trained on nine tenths of TRAIN and early-stopped on the remaining tenth (monitor 0.4577 at epoch 21). A further pass over all of TRAIN at learning rate 5e-5 scored DEV 0.4548 and was not kept. DEV of the shipped file, reported only: 0.4550.

```
python scripts/train_neural.py --kind pointwise --train data/train --dev data/dev --monitor train-slice \
  --hidden 64 --epochs 20 --patience 3 --out-dir submission
python scripts/train_neural.py --kind pairwise --train data/train --dev data/dev --monitor train-slice \
  --pair-weighting delta_ndcg --pair-sampling gain --hidden 256 256 256 --queries-per-batch 64 \
  --pairs-per-query 4000 --lr 5e-4 --lr-decay 0.95 --scale winsor --epochs 40 --patience 8 \
  --out-dir submission
```

The same run with patience 8 and uniform pairs scored DEV 0.4531. Gain sampling plus weight decay 1e-5 scored DEV 0.4551. A second seed of the shipped schedule scored DEV 0.4532. Averaging three seeds (0.3459), a gain-target term (0.4102), and a listwise softmax (0.3823) were not used. Input noise with 3% feature dropout scored DEV 0.4523. Drawing pairs by gain times error scored DEV 0.4536. Neither replaced the shipped file.

`pointwise_model.pt` is a smaller MLP, hidden width 64, squared error, same early-stop split of TRAIN. It is not used to initialize the pairwise model. DEV, reported only: nDCG@10 0.4225.

Dev-set nDCG@10 of the three submitted models: pointwise 0.4225, pairwise 0.4550, LambdaMART 0.7079. The LambdaMART number is in-training, because DEV is inside that fit. Pointwise and pairwise were trained on TRAIN and then scored on DEV.

## Recipe
`recipe.py` reads each benchmark's own train and vali files. Values above 1e30 are stored as missing. Leaf size, column sampling, L2, and the round budget follow the number of rows and features. It early-stops on that benchmark's vali nDCG@10, then refits on train+vali when the size-based cost estimate fits in the time budget. A wall-clock stop at 15 minutes is a safety limit inside the 20-minute cap. The fit uses each query's raw features, within-query percentiles, and within-query z-scores. `rerank_with_ltr` rebuilds those columns. On MSLR TRAIN slices, including columns that were rescaled and slices with added noise columns, the three-column form raised the query-held-out nDCG@10 over raw features alone.

## Generalization
The shipped tree uses 127 leaves and 2835 trees, with DEV inside the fit, so 0.7079 is an in-training score. The neural models were not refit on DEV. Fusion weights were chosen on TRAIN. The recipe's hyperparameters come from the benchmark it is handed, not from MSLR constants.

## AI-use disclosure
Claude Code (Anthropic, Claude Opus 5.5) and Cursor (Grok) were used to read the spec and starter code, write the submission modules, change the training scripts, run the experiments, and draft these notes.

## Code provenance
No external code was copied. Formulas, from the assignment references: RRF (Cormack, Clarke and Buettcher, SIGIR 2009); CombSUM and CombMNZ (Fox and Shaw, TREC-2 1994); RankNet (Burges et al., ICML 2005); LambdaRank |ΔnDCG| pair weights (Burges, MSR-TR-2010-82) in `delta_ndcg_weights` and `weighted_ranknet_loss`. LambdaMART is LightGBM `lambdarank`. Libraries: NumPy, LightGBM, PyTorch, as in `requirements.txt`.

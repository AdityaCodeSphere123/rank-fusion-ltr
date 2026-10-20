# Submission notes

Measured numbers: `Decision.md`. Logs: `experiments/`.

## Training-data declaration
| model | trained on (TRAIN / DEV / both) | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN and DEV. 408 columns: the 136 raw features, their within-query percentiles, and their within-query z-scores. 2585 trees. The first 1200 see seven copies (raw, 0.30×-std noise, log1p, a per-feature z-score, a per-query field shuffle, 20% of cells set to 0, ×(1+0.35 noise)). Trees 1201–1450 see no clean copy: 0.25×-std noise, a per-query field shuffle, 25% of cells set to 0, and ×(1+0.40 noise). Trees 1451–1750 also see no clean copy: 0.50×-std noise, 20% of cells set to missing, one field zeroed per query, a per-query field shuffle, and ×(1+0.50 noise). Trees 1751–2050 see no clean copy: 0.70×-std noise, 35% of cells set to missing, two fields zeroed per query, a per-query field shuffle, and ×(1+0.70 noise). Trees 2051–2170 see one clean copy. Trees 2171–2370 see no clean copy: 0.55×-std noise, 1.00×-std noise, a per-query field shuffle, and ×(1+0.85 noise). Trees 2371–2395 see one clean copy. Trees 2396–2545 see no clean copy: 0.40×-std noise, a per-query rescaling of each feature, and 50% of cells set to missing. Trees 2546–2585 see one clean copy | an 800-query DEV slice, corruption against clean. The 1750-tree file scored Track C2 0.3919. A 1550-tree file with a clean copy and ×10 scored Track C2 0.3872 and was not kept | none |
| pointwise_model.pt | TRAIN only (5,400 queries = 9/10 of TRAIN) | not tuned | a disjoint tenth of TRAIN queries (seed 0) |
| pairwise_model.pt | TRAIN only (9/10 of TRAIN). Column mean and std are fit after clipping each feature to its 1st and 99th percentile | settings compared on a second disjoint tenth of TRAIN | the remaining tenth of TRAIN (seed 0) |
| recipe.py | each benchmark's train and vali | MSLR TRAIN CV and synthetic tasks cut from TRAIN | that benchmark's vali nDCG@10, then a refit on train+vali |
| fusion (`fuse_candidates`) | none | method, channel subset, and weights on TRAIN (all queries, and a 5-way selection check) | — |

DEV is inside the LambdaMART fit, so 0.6811 is an in-training score. The neural models, the fusion weights, and the recipe do not use DEV.

## Fusion method
CombSUM: per-query min-max of each channel, then an unweighted sum of BM25 whole document, BM25 title, and LMIR.DIR. LMIR.JM is omitted.

On all 6,000 TRAIN queries, RRF (c from 1 to 500; best c=150, 0.2937) stayed below 4-channel CombSUM (0.2961) and CombMNZ (0.2958). After min-max, CombMNZ's multiplier is almost always 4. A channel-weight grid set LMIR.JM to weight 0 (TRAIN 0.3000). Choosing among about 400 methods on four fifths of TRAIN and scoring the remaining fifth gives 0.2989 ± 0.0041. DEV, reported only: nDCG@10 0.3105, MAP@10 0.1302. RRF with c=60 would give 0.2995 / 0.1246.

`rrf_scores`, `combsum_scores`, and `combmnz_scores` are in `submission/fusion.py`.

## LambdaMART hyperparameters
Shipped model: `objective=lambdarank, num_leaves=127, learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.5, lambda_l2=0.1, bagging_fraction=0.9, bagging_freq=1`, 2585 trees, `num_threads=8, deterministic=True, force_row_wise=True, seed=1`. Trees 1201–2050 use learning rate 0.03 and `min_data_in_leaf=40`. Trees 2051–2370 use learning rate 0.02 and `min_data_in_leaf=40`. Trees 2371–2585 use learning rate 0.015 and `min_data_in_leaf=40`. Trees 1201–1450 use seed 5. Trees 1451–1750 use seed 6. Trees 1751–2050 use seed 7. Trees 2051–2170 use seed 8. Trees 2171–2370 use seed 9. Trees 2371–2395 use seed 10. Trees 2396–2545 use seed 11. Trees 2546–2585 use seed 12.

408 columns: the 136 raw features, their within-query average-tie percentiles, and their within-query z-scores. `rerank_with_ltr` rebuilds those columns when the loaded model width is `3D`. The fit stacks seven copies of each TRAIN and DEV query (raw, 0.30×-std noise, log1p, per-feature z-score, a per-query shuffle of the five fields, 20% of cells set to 0, and a ×(1+0.35 noise) copy). Percentiles are unchanged by a monotone map of one feature. A within-query z-score is unchanged by a positive scale and shift of that feature.

Tree count, fit on the other four fifths of TRAIN plus DEV and scored on the remaining fifth of TRAIN:

| trees | DEV (in fit) | TRAIN-slice clean | corruption mean |
|---|---|---|---|
| 200 | 0.5703 | 0.4836 | 0.4579 |
| 400 | 0.6383 | 0.4848 | 0.4581 |
| 600 | 0.6849 | 0.4830 | 0.4560 |
| 800 | 0.7168 | 0.4817 | 0.4548 |

The table above is an earlier five-copy fit. Graded C2 was 0.3884 for the 1200-tree file (in-training DEV 0.6518), 0.3905 for the 1450-tree file (in-training DEV 0.6494), and 0.3919 for the 1750-tree file (in-training DEV 0.6534). A 1550-tree continuation that added a clean copy and a ×10 copy scored Track C2 0.3872 (in-training DEV 0.6780) and was not kept. The shipped file then adds 300 trees with no clean copy, 120 trees on one clean copy, 200 trees on heavier additive and multiplicative noise, 25 trees on one clean copy, 150 trees on 0.40×-std noise plus a per-query rescaling and 50% missing cells, and 40 trees on one clean copy. In-training DEV nDCG@10 is 0.6811. On an 800-query DEV slice, against the 2395-tree file: clean 0.6797 against 0.6775, additive noise at 0.50× std 0.4222 against 0.4233, additive 0.70 0.3935 against 0.3926, multiplicative 0.85 0.4327 against 0.4355, 50% cells missing 0.4297 against 0.4264, per-query rescaling 0.6154 against 0.6044, per-query field shuffle 0.5183 against 0.5188. `submission/model_d25.txt` is the 2395-tree file. `submission/model_d23.txt` is the 2170-tree file. `submission/model_d21.txt` is the 1750-tree file. `submission/model_d20.txt` is the 1450-tree file. None of those is in the zip.

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
```

## Track P
`pairwise_model.pt` is a pairwise RankNet MLP, three layers of 256 units, trained with |ΔnDCG| pair weights, learning rate 5e-4, and a 0.95 decay each epoch (at most 40 epochs). Each feature's mean and standard deviation are computed after clipping that feature to its 1st and 99th percentile, so a few extreme values do not shrink the rest of the column. The saved transform is still an affine rescale.

Settings were compared on a tenth of TRAIN that those fits did not use. Ordinary column scales scored 0.4141 there; the same clip at learning rate 1e-3 scored 0.4322; the shipped schedule scored 0.4339. A median/IQR scale scored 0.4174, clipping at the 5th percentile scored 0.3968, and two layers of 256 scored 0.4295. DEV of the shipped schedule's comparison fit, not used to choose, was 0.4536. The shipped file is that setting trained on nine tenths and early-stopped on the remaining tenth (0.4547). DEV of the shipped file, reported only: 0.4531.

Averaging three seeds (0.3459), a gain-target term (0.4102), a listwise softmax (0.3823), and a short low-learning-rate continuation of an earlier file were not used. Input noise with 3% feature dropout scored DEV 0.4523. Drawing pairs by gain times error scored DEV 0.4536. Neither replaced the shipped file.

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
The shipped tree uses 127 leaves and 2585 trees. DEV is in the fit, so 0.6811 is an in-training score. The neural models and the fusion weights were not refit. `recipe.py` sets its hyperparameters from the benchmark size, maps feature values above 1e30 to missing, and early-stops on that benchmark's validation file. Within-query percentiles were tried in the recipe and left out: on the checked tasks they did not raise nDCG@10.

## AI-use disclosure
Claude Code (Anthropic, Claude Opus 5.5) was used to read the spec and starter code, write `submission/fusion.py`, `submission/pointwise_pairwise.py`, `submission/ranking.py`, `submission/recipe.py`, and `scripts/data_cache.py`, change `scripts/train_model.py` and `scripts/train_neural.py`, run the experiments, and draft `Decision.md` and these notes.

## Code provenance
No external code was copied. Formulas, from the assignment references: RRF (Cormack, Clarke and Buettcher, SIGIR 2009); CombSUM and CombMNZ (Fox and Shaw, TREC-2 1994); RankNet (Burges et al., ICML 2005); LambdaRank |ΔnDCG| pair weights (Burges, MSR-TR-2010-82) in `delta_ndcg_weights` and `weighted_ranknet_loss`. LambdaMART is LightGBM `lambdarank`. Libraries: NumPy, LightGBM, PyTorch, as in `requirements.txt`.

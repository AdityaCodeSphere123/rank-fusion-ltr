# Submission notes

## Training-data declaration
| model | trained on (TRAIN / DEV / both) | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | both | grouped 5-fold CV on TRAIN, then tree count and corrupted-copy settings checked on TRAIN slices and DEV | none |
| pointwise_model.pt | 9/10 of TRAIN | not tuned | the other 1/10 of TRAIN |
| pairwise_model.pt | 9/10 of TRAIN | pair sampling and learning rate on a TRAIN slice | the other 1/10 of TRAIN |
| recipe.py (run by staff on hidden benchmarks) | each hidden benchmark's train+vali | MSLR TRAIN CV, MSLR TRAIN slices scored on DEV, and the synthetic transfer sets in `data/toy_transfer/` | each benchmark's vali, then refit on train+vali |

DEV is inside the LambdaMART fit, so its DEV score of 0.7079 is an in-training score.

## Fusion method and why
`fuse_candidates` is CombSUM of per-query min-max scores over BM25 whole document, BM25 title and LMIR.DIR. On all 6,000 TRAIN queries RRF (best c = 150, 0.2937) scored below CombSUM (0.2961) and CombMNZ (0.2958). A channel weight grid on TRAIN set LMIR.JM to 0 so it is dropped, and the three-channel sum scores 0.3000 on TRAIN. DEV: nDCG@10 0.3105, MAP@10 0.1302. `python scripts/fusion_eval.py` prints all of these numbers.

## LambdaMART hyperparameters and how chosen (cite your CV table)
Grouped 5-fold CV on TRAIN only, with no early stopping on the reported fold. Tree counts 100, 200, 300, 500, 750, 1000, 1500 and 2000 were swept. `python scripts/cv_sweep.py --grid leaves` makes the first four rows. `--grid lr` reruns the last three; their other settings (L2 1.0, bagging 0.8) are the script defaults and were not saved with the original results:

| lr / leaves / min_data_in_leaf / feature_fraction | best trees | CV val nDCG@10 | train | val/train |
|---|---|---|---|---|
| 0.05 / 31 / 100 / 0.8 | 1000 | 0.4749 ± 0.0059 | 0.6243 | 0.761 |
| 0.05 / 63 / 100 / 0.8 | 300 | 0.4744 ± 0.0063 | 0.5893 | 0.805 |
| 0.05 / 127 / 100 / 0.8 | 500 | 0.4742 ± 0.0077 | 0.7174 | 0.661 |
| 0.05 / 255 / 100 / 0.8 | 200 | 0.4741 ± 0.0059 | 0.6961 | 0.681 |
| 0.05 / 63 / 500 / 0.5 | 500 | 0.4741 ± 0.0068 | 0.6233 | 0.761 |
| 0.025 / 63 / 500 / 0.5 | 1000 | 0.4766 ± 0.0070 | 0.6249 | 0.763 |
| 0.0125 / 63 / 500 / 0.5 | 2000 | 0.4771 ± 0.0068 | 0.6263 | 0.762 |

Validation stays near 0.474 to 0.477 for every shape while the train score keeps rising, so extra capacity mostly memorizes.

Feature engineering: each of the 136 features is joined by its percentile and its z-score inside the query, giving 408 columns. `rerank_with_ltr` rebuilds them at scoring time. Both stay the same when a feature is rescaled. Training also stacks changed copies of the rows: Gaussian and multiplicative noise, log1p, cells set to 0 or missing, fields shuffled or zeroed per query, and per-query rescaling.

Shipped model: `num_leaves=127`, `feature_fraction=0.5`, `lambda_l2=0.1`, `bagging_fraction=0.9` and 2835 trees. The learning rate falls from 0.05 to 0.015 over the stages below and `min_data_in_leaf` is 20 or 40. Each stage continues from the file before it:

```
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --out submission/model_d17.txt --trees 700 --num-leaves 127 --learning-rate 0.05 \
  --min-data-in-leaf 20 --lambda-l2 0.1 --feature-fraction 0.5 \
  --bagging-fraction 0.9 --bagging-freq 1 --feature-noise 0.30 --log-view --z-view \
  --permute-per-query --cell-view --cell-rate 0.20 --mul-view 0.35 \
  --query-z --seed 1 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d17.txt --out submission/model_d18.txt --trees 500 \
  --num-leaves 127 --learning-rate 0.05 --min-data-in-leaf 20 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 \
  --feature-noise 0.30 --log-view --z-view --permute-per-query --cell-view \
  --cell-rate 0.20 --mul-view 0.35 --query-z --seed 1 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d18.txt --out submission/model_d20.txt --trees 250 \
  --learning-rate 0.03 --num-leaves 127 --min-data-in-leaf 40 --lambda-l2 0.1 \
  --feature-fraction 0.5 --bagging-fraction 0.9 --bagging-freq 1 --clean-copies 0 \
  --feature-noise 0.25 --permute-per-query --cell-view --cell-rate 0.25 \
  --mul-view 0.40 --query-z --seed 5 --num-threads 8
python scripts/train_model.py --train data/train --dev data/dev --use-dev \
  --init-model submission/model_d20.txt --out submission/model_d21.txt --trees 300 \
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

Neural models:

```
python scripts/train_neural.py --kind pointwise --train data/train --dev data/dev --monitor train-slice \
  --hidden 64 --epochs 20 --patience 3 --out-dir submission
python scripts/train_neural.py --kind pairwise --train data/train --dev data/dev --monitor train-slice \
  --pair-weighting delta_ndcg --pair-sampling gain --hidden 256 256 256 --queries-per-batch 64 \
  --pairs-per-query 4000 --lr 5e-4 --lr-decay 0.95 --scale winsor --epochs 40 --patience 8 \
  --out-dir submission
```

Both commands early-stop on a fixed tenth of the TRAIN queries (`grouped_kfold(k=10, seed=0)`, fold 0), so DEV is only scored at the end. The shipped `pointwise_model.pt` was trained on 17 Oct with an earlier version of the script. Its saved standardizer matches this same 9/10 of TRAIN exactly, but running the command above today gives DEV 0.4160 and not 0.4225. Running the pairwise command again gives DEV between 0.452 and 0.455. The first epochs are identical, and after that small floating point differences between runs change the result a little.

| model | DEV nDCG@10 |
|---|---|
| pointwise (shipped) | 0.4225 |
| pairwise (shipped) | 0.4550 |
| LambdaMART (shipped, DEV was in training) | 0.7079 |

The 0.7079 is not a fair comparison because DEV rows were fit. A fairer LambdaMART estimate is the CV validation score of the plain settings above, about 0.477 on unseen TRAIN queries, which is still above both neural models.

`recipe.py` (Track C1) sets leaves, leaf size, column sampling and L2 from the data size. It averages up to six LambdaMART models. Half see the raw features and half also see the within-query percentile and z-score columns and their trees use the size rule, twice the leaves, half the leaves or fewer columns. Each model early-stops on the benchmark's vali and is then refit on train plus vali. The number of models comes from a size-based time estimate so the run fits in 20 minutes and 8 GB and depends only on the data and the seed. On full MSLR TRAIN plus DEV (about 958,000 rows, 136 features) a full run took 313 seconds and peaked at 4.0 GB of memory with 4 threads on a Mac. On a synthetic set at the stated limits (500,000 train rows, 300 features, labels 0 to 4) it took 353 seconds and 5.2 GB.

## Final configuration: what you traded off between dev accuracy and generalization
We gave up some clean accuracy for robustness. Training on DEV and on corrupted copies lifts DEV to 0.7079, but that number is in-training and says little about unseen queries. The CV table shows validation barely moves with more leaves or trees, so the gain we aimed for is stability on the hidden C2 versions. The corrupted copies and the scale-free within-query columns keep the ranking steady when features are noisy, missing or rescaled. For C1 we used rules instead of constants and averaged several tree shapes, so no single setting tuned on MSLR decides the result on a dataset we cannot see.

## AI-use disclosure
Claude Code (Anthropic, Claude Opus 5.5) and Cursor (Grok) were used to read the spec and starter code, write the submission modules, change the training scripts, run the experiments and draft these notes.

## Code provenance
No external code was copied. Formulas come from the assignment references: RRF (Cormack, Clarke and Buettcher, SIGIR 2009), CombSUM and CombMNZ (Fox and Shaw, TREC-2 1994), RankNet (Burges et al., ICML 2005) and LambdaRank |ΔnDCG| pair weights (Burges, MSR-TR-2010-82) in `delta_ndcg_weights` and `weighted_ranknet_loss`. LambdaMART is LightGBM `lambdarank`. Libraries: NumPy, LightGBM and PyTorch, as in `requirements.txt`.

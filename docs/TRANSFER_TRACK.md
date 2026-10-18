# Track C1: transfer to undisclosed benchmarks (35% of the grade)

Staff will run your training recipe on **two undisclosed learning-to-rank benchmarks** and score the resulting models on each benchmark's held-out split. They are not MSLR, and a model trained on MSLR cannot be applied to them: what transfers is your *method*. We do not say which datasets they are, how many features they have, or what the features mean.

## What you submit
`submission/recipe.py` with `train(train_path, vali_path, out_model_path, seed=0)` (the template is a working fixed-hyperparameter LambdaMART). Staff call it once per benchmark in an isolated directory, load the model it writes with your own `prepare()`, and score `rerank_with_ltr()` on that benchmark's held-out split. Each benchmark's nDCG@10 is percentile-ranked separately and the two percentiles are averaged.

## What you can rely on
* Flat-feature SVMLight files (`label qid:N 1:v ... D:v`); trailing `# comments` are ignored; all files of one benchmark share the same D.
* D is unknown but at most about 300; labels are integers 0..4 (a benchmark may use fewer levels); at most about 500,000 training rows (staff subsample whole queries if a benchmark is larger).
* Feature order, scales and meaning are not given and may be arbitrary; do not hard-code feature indices.
* 20 minutes wall-clock per benchmark on 4 CPU threads; no network; read only the two input files; deterministic given the seed.
* A recipe that raises, times out, or writes a model your `prepare()` cannot load scores 0 on that benchmark. `rerank_with_ltr()` must work for any feature dimension. `fuse_candidates()` is not used.

## Develop locally
`python scripts/run_recipe.py` runs your recipe on the three synthetic transfer sets in `data/toy_transfer/` (40, 90 and 250 features; labels 0-4 or 0-2) and reports vali nDCG@10 and training time. They are synthetic stand-ins: passing them only shows that the recipe is dimension-agnostic, label-agnostic, fast and deterministic.

## What tends to transfer
Choices expressed as rules rather than constants (tree count and `min_data_in_leaf` scaled to data size, `feature_fraction` for wide inputs), and a small cross-validation inside the time limit. Constants tuned to MSLR often over- or under-fit a dataset of a different size or shape.

## Integrity
Do not try to identify the benchmarks or consult the labels of any public learning-to-rank dataset for this assignment; doing so is an integrity violation, like consulting a solution key (spec Sections 6 and 10).

# A3 -- Rank Fusion + LTR Arena (student starter)

Fuse four base rankers, then train pointwise / RankNet / LambdaMART rankers on MSLR-WEB10K Fold1 without overfitting. Full spec: the assignment PDF. Held-out data is **not** here and must not be sought (honor code).

## Quick start
```
pip install -r requirements.txt
bash scripts/smoke_test.sh                 # toy data, trains, runs the harness, runs tests
python scripts/prepare_mslr.py --fold-dir <MSLR-WEB10K/Fold1>     # real data -> data/train, data/dev
python scripts/run_recipe.py                      # Track C1 dry run on toy data (synthetic transfer sets; see docs/TRANSFER_TRACK.md)
python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 400   # local train-vs-validation gap
python scripts/train_model.py --train data/train --out submission/model.txt --trees 200
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt
```

## What you write
| file | what |
|---|---|
| `submission/fusion.py` | `rrf_scores` and `combsum_scores` and/or `combmnz_scores` (TODOs). Until RRF works, `ranking.py` falls back to sorting by the BM25 channel |
| `submission/pointwise_pairwise.py` | `set_seed`, `pointwise_loss`, `make_pairs`, `ranknet_loss`, the two `*_step`s, the training loop (TODOs). Your trained `pairwise_model.pt` is also scored on the leaderboard (Track P): see the module docstring for what you may tune |
| `submission/ranking.py` | the competition entry: keep the shape, improve the internals |
| `scripts/train_model.py` | your reproducible LambdaMART training script |
| `submission/recipe.py` | your training recipe, run by staff on undisclosed benchmarks for Track C1 (working default provided; tune it so it transfers) |

Self-check your parts: `python -m pytest tests/test_student_todos.py` (fails until TODOs are done). CI only gates the interface (`tests/` minus that file).

Docs: `docs/SUBMISSION_INTERFACE.md`, `docs/GRADING.md`, `docs/RANKNET_PRIMER.md`, `docs/DATA.md`, `docs/TRANSFER_TRACK.md`, `docs/SUBMISSION_NOTES.md` (fill in and keep as your README section).

## Layout
`harness/` metrics, I/O, CV, leaderboard formulas, runner (do not edit; staff run an identical copy) - `submission/` your entry - `scripts/` tooling - `data/toy/` and `data/toy_transfer/` synthetic data - `tests/`.

The shipped `submission/model.txt` / `pointwise_model.pt` / `pairwise_model.pt` are trained on the toy data only: they exist so the pipeline runs end to end.

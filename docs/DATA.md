# Data

MSLR-WEB10K Fold1 (Qin & Liu, 2013). Line format: `label qid:N 1:v1 2:v2 ... 136:v136` (labels 0-4, features 1-based). No document ids -- the harness uses `"<qid>:<row>"`.

| split | MSLR file | location | labels |
|---|---|---|---|
| TRAIN | `train.txt` | `data/train/` | released |
| DEV | `vali.txt` | `data/dev/` | released; public leaderboard |
| HELD-OUT | `test.txt` | withheld | **never obtain or consult** (honor code) |

Get the data: download MSLR-WEB10K.zip from Microsoft Research yourself, then `python scripts/prepare_mslr.py --fold-dir <.../Fold1>`. For fast iteration use `data/toy/` (synthetic).

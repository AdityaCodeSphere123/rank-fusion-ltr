"""
harness/recipe_eval.py -- Track C1 machinery (assignment Sections 4.1, 7).

run_recipe_and_evaluate() calls the team's submission/recipe.py train() on a (train, vali) pair of flat-feature
files, then loads the model it wrote through the team's OWN submission/ranking.py (prepare + rerank_with_ltr,
no base-ranker channels) and returns nDCG@10 on an evaluation file. Staff run it in a subprocess inside an
isolated directory with a wall-clock limit; students run it locally via scripts/run_recipe.py on the synthetic
transfer sets in data/toy_transfer/ (the hidden benchmarks' held-out splits are staff-only).
"""
import time
from typing import Dict, Optional

from harness.feature_io import read_svmlight_ltr, to_qrels
from harness.metrics import evaluate_run
from harness.run_harness import rank_dataset


def run_recipe_and_evaluate(train_path: str, vali_path: str, eval_path: str, out_model_path: str,
                            seed: int = 0, k: int = 10) -> Dict[str, float]:
    import submission.ranking as ranking
    import submission.recipe as recipe

    t0 = time.perf_counter()
    recipe.train(train_path, vali_path, out_model_path, seed)
    train_seconds = time.perf_counter() - t0

    dataset = read_svmlight_ltr(eval_path)
    ranking.prepare(out_model_path)
    _, run = rank_dataset(dataset, None, ranking.rerank_with_ltr, k=k, channels={})
    agg = evaluate_run(run, to_qrels(dataset))["aggregate"]
    return {"ndcg@10": agg["ndcg@10"], "map@10": agg["map@10"], "train_seconds": train_seconds,
            "num_eval_queries": len(dataset)}

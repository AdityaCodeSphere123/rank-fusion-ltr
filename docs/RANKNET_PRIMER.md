# RankNet primer (no neural-network background assumed)

**The network.** A function `score = f(features)`: 136 numbers in, one number out. Learning = nudging its weights so good documents get higher scores than bad ones.

**One training step** (repeated thousands of times):
1. *Forward pass*: feed a minibatch of feature rows through the network to get scores.
2. *Loss*: one number measuring how wrong the scores are. Pointwise: `mean((score - label)^2)`. Pairwise (RankNet): for a pair where document i is more relevant than j, `log(1 + exp(-(s_i - s_j)))` -- near 0 when `s_i` is far above `s_j`, large when the order is wrong.
3. *Backward pass* `loss.backward()`: autograd computes, for every weight, how the loss changes if the weight moves.
4. *Optimizer step* `optimizer.step()`: move each weight a small step (learning rate) downhill.
5. `optimizer.zero_grad()`: clear gradients, otherwise they accumulate across batches.

**Tiny worked example.** Pair scores `s_i = 2, s_j = 0` -> loss `log(1+e^-2) = 0.1269`. Pair scores `0, 0` -> `log 2 = 0.6931`. Mean over both = **0.410038**. Pair scores `-1000, 1000` -> loss `2000`: compute with `softplus(-(s_i - s_j))` / `torch.logaddexp(0, -d)` so `exp(2000)` never overflows.

**Why pairs only within a query:** relevance is relative to a query. Comparing a document of query A with one of query B means nothing.

**Failure modes**
* Unscaled features -> loss does not move. Use the provided `Standardizer` (fit on TRAIN only).
* Learning rate too high -> loss jumps or goes `nan`; try 1e-3, then 3e-4.
* Pairs formed across queries.
* Not reshuffling each epoch -> one query's pairs sit together and bias the gradient.
* Forgetting `zero_grad()`.
* Not seeding: `random`, `numpy`, `torch` all need seeding.
* Judging by training loss: loss is a proxy; watch dev nDCG@10 each epoch. Evaluate with `network.eval()` and `torch.no_grad()`, scoring whole queries.

**Early stopping.** Keep the weights from the epoch with the best dev nDCG@10; stop after `patience` epochs with no improvement.

**Beyond plain RankNet (Track P).** RankNet treats every pair the same, but a swap near the top of the ranking changes nDCG@10 far more than one near the bottom. LambdaRank (Burges, 2010) multiplies each pair's loss by |delta nDCG|, the change in nDCG@10 if the two documents swapped places under the current scores. You may use this, or smarter pair sampling, for your leaderboard model -- in a separate function. The graded `ranknet_loss` stays unweighted.

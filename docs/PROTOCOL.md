# Protocol

## Tasks

For every user the check-in history is sorted by time and reduced to the sequence
of distinct POIs in order of first visit.

* Support set: the first `k` of the first `k_max` (10) distinct POIs.
* Query set: the next `q` (20) distinct POIs, i.e. POIs `k_max+1 ... k_max+q`.
  Because the query always starts after `k_max`, varying `k` never changes the
  prediction target.
* Friends: up to `n_friends` (3) friends who are training users, each represented
  by their own first `k_max` POIs.
* Splits are user-disjoint, so every test user is a genuine cold-start user.

## Shared representation

POI embedding: a learned ID embedding is concatenated with an MLP encoding of the
content vector (normalised latitude and longitude, one-hot category, optionally a
standardised log-popularity), fused by a linear layer with ReLU and L2-normalised
(64 dimensions).

User embedding: mean of the support POI embeddings, followed by a two-layer
network and L2 normalisation. The friend prototype is encoded the same way and
blended with weight `social_w` (0.25).

Score: cosine similarity. Loss: softplus (logistic) loss on scores divided by the
temperature `tau` (0.05), with clipping to [-10, 10]; positives are query POIs,
negatives are sampled unvisited POIs.

## Learners

* `proto`: the user embedding is computed directly from the support set; one
  forward pass, no inner loop. Gradient-norm clip 1.0.
* `maml`: one inner gradient step on the support set (context / inner-positive
  split, element-wise gradient clipping, inner rate `alpha_in`), then the query
  loss is computed with the adapted weights and back-propagated through the inner
  step (second order). Gradient-norm clip 0.5.

Both use Adam, batch of 16 tasks, early stopping on validation NDCG@10 with
patience 3.

## Evaluation

For each test user the query positives and a set of negatives are ranked together;
NDCG@K and Recall@K are computed for K in {10, 20}, with the ideal DCG based on
`min(q, K)` relevant items.

* `global`: `q * neg_per_pos` = 2,000 negatives drawn uniformly from POIs the user
  has never visited.
* `local`: 2,000 negatives drawn from the `local_pool` (5,000) unvisited POIs that
  are closest to the centroid of the support set.
* `full`: all unvisited POIs.

Negatives depend only on the seed and the user, are cached in the output
directory (`negs_k<k>_s<seed>.pkl`) and are therefore identical for every model.

Ties: positives are placed before negatives and sorted with a stable sort, so an
exact tie favours the positive. For continuous cosine scores this has no effect;
for the popularity baseline ties are broken by a negligible random perturbation.

## Statistical testing

`scripts/aggregate.py` performs paired two-sided Wilcoxon signed-rank tests over
test users (per-user NDCG@10) for a fixed list of system pairs, and reports mean
and standard deviation over seeds for every configuration.

# Cold-start POI recommendation: prototypical networks versus MAML

Reference implementation of a controlled comparison between two meta-learning
strategies for **user cold-start point-of-interest (POI) recommendation** on
Foursquare check-ins. Every user is a task: the first `k` visited POIs form the
support set and the next `q` POIs form the query set. Both learners are trained
on one shared representation, with the same tasks, the same loss and the same
evaluation negatives, so that any difference is attributable to the learning
strategy and not to the surrounding pipeline.

Authors: Azadeh Tabatabaei, Aran Olfat Shayan (University of Science and Culture).

## Branches

| Branch | Content |
|---|---|
| `main` | Code of the defended MSc thesis, kept unchanged for traceability (tag `thesis-defended`). |
| `paper` | This code base: unified implementation, extended evaluation and a reproducible pipeline for the journal paper. |

## Repository layout

```
src/csrec/            installable-by-path Python package
  config.py             all hyper-parameters and the command line interface
  data.py               data loading and construction of per-user tasks
  geo.py                haversine distance
  negatives.py          fixed, cached evaluation negatives (global and local)
  metrics.py            NDCG@K / Recall@K and the evaluation loop
  baselines.py          popularity and distance-discounted popularity
  model.py              shared POI/user representation (functional, MAML-ready)
  train.py              meta-training for the prototypical and MAML learners
  run.py                entry point (python -m csrec)
scripts/
  prepare_data.py       raw Foursquare files -> processed files (checkins, pois, splits, social)
  compare_processed.py  agreement report between two processed directories
  make_sample.py        builds a small sample from the full processed data
  run_experiments.sh    the experiment grid (quick and full profiles)
  aggregate.py          mean/std tables, cost table, Wilcoxon tests
tests/                unit tests for metrics, geometry and task construction
data/sample/          bundled sample (2,100 users, about 5 MB), ready to run
data/README.md        data format and how to obtain the full data
docs/PROTOCOL.md      task construction, models and evaluation protocols
reference_results/    per-run JSON files of the Japan experiments in the paper
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src          # or run everything through the Makefile
```

Verify the setup:

```bash
make test
```

## Quick start (20 to 30 minutes on a laptop CPU)

The repository ships with a small sample so the whole pipeline can be executed
immediately. Results on the sample are only meant to check that everything runs;
they are not the numbers reported in the paper.

```bash
make quick                      # equivalent to scripts/run_experiments.sh quick
cat results/quick/summary.md    # tables produced by scripts/aggregate.py
```

A single run:

```bash
python -m csrec --data data/sample --learner proto --seed 1 --epochs 3
python -m csrec --data data/sample --learner maml  --seed 1 --epochs 3
python -m csrec --data data/sample --learner pop
python -m csrec --help          # every option with its default
```

## Reproducing the paper

1. Download the three raw Foursquare files into `data/raw/` and build the processed
   Japan data (files are streamed, so memory use is modest; see `data/README.md`):

   ```bash
   python scripts/prepare_data.py --raw data/raw --out data/processed --countries JP
   ```

   The numbers in `reference_results/` were produced from the files used in the
   thesis. `prepare_data.py` rebuilds them with identical users, indices, categories
   and social graph, and 99.996% of the check-ins, but the stratified random split
   is not byte-identical (see `data/README.md`). Expect small differences in the
   last digits when re-running on a freshly prepared split.
2. Run the grid. Three seeds for the main systems, one seed for the ablations:

   ```bash
   DATA=data/processed make full          # hours on CPU, much faster on a GPU
   DEVICE=cuda DATA=data/processed make full
   ```

3. Collect the tables:

   ```bash
   python scripts/aggregate.py --results results/full
   ```

Each run is identified by a tag `<learner>_<variant>_k<k>_s<seed>[_a<alpha>][_mixneg]`
and writes `<tag>.json` (arguments, validation history, test metrics, timing) and
`<tag>_peruser.pkl` (per-user metrics used for the paired significance tests).

### Options that define the experiments

| Option | Values | Meaning |
|---|---|---|
| `--learner` | `proto`, `maml`, `pop`, `localpop` | meta-learner or non-learned baseline |
| `--ablation` | `none`, `no_geo`, `no_social`, `no_id`, `pop_feat`, `melu` | variant of the shared representation |
| `--k` | integer, default 10 | support size (query stays fixed) |
| `--train_neg` | `uniform`, `mixed` | `mixed` draws half of the training negatives from the 5,000 nearest unvisited POIs |
| `--alpha_in` | float | MAML inner learning rate |
| `--seed` | integer | model seed and evaluation-negative seed |
| `--device` | `cpu`, `cuda`, `mps` | computation device |

## Evaluation protocols

Every model is scored under three candidate sets, always against the same
negatives for a given seed (details in `docs/PROTOCOL.md`):

* `global`: 100 uniformly sampled unvisited POIs per query positive (the protocol
  of the original thesis).
* `local`: negatives sampled from the 5,000 unvisited POIs closest to the user's
  support centroid, which removes the trivial "wrong city" negatives.
* `full`: ranking against every unvisited POI, with no sampling.

Sampled metrics can change the ranking of systems (Krichene and Rendle, KDD 2020),
which is why all three are reported.

## Reference results (Japan, full data, three seeds)

NDCG@10, mean over seeds. Files are in `reference_results/japan/`.

| System | global | local | full |
|---|---|---|---|
| Popularity | 0.333 | 0.340 | 0.085 |
| Local popularity | 0.296 | 0.101 | 0.065 |
| MAML | 0.377 | 0.253 | 0.078 |
| Prototypical | 0.379 | 0.258 | 0.075 |
| Prototypical + popularity feature | 0.409 | 0.286 | 0.092 |

Cost on the same CPU: about 105 s per epoch for the prototypical learner and
about 480 s for MAML (4.6x), and 0.013 s versus 0.096 s to score one user (7.4x).

## Differences from the thesis implementation

The thesis code is preserved on `main`. The code here changes the following, and
the paper numbers must not be mixed with thesis numbers:

1. All models use the same query size (thesis: 15 for the prototypical model and
   25 for the MAML-based models).
2. Support sets are chronological (the earliest unique POIs), not random.
3. The social component encodes each friend from the friend's own support POIs.
   The thesis prototypical code passed friend user IDs where POI IDs were expected.
4. Both learners share one representation and one loss. In the thesis the
   prototypical model used a softmax contrastive loss and no user encoder.
5. Hard-negative sampling is not part of the model. The thesis code built a
   category map but never used it, so the corresponding ablation row was
   identical to the full model.
6. Evaluation adds the `local` and `full` candidate sets and a popularity
   reference, on top of the thesis `global` protocol.

## Known limitations

* Only the Japan subset has been run under this protocol. `prepare_data.py` already
  supports the Asia country list (`--countries JP,KR,SG,TW,HK,CN`), but those runs
  have not been executed.
* MeLU, MAMO and TaNP are not yet implemented under the unified protocol
  (`--ablation melu` is a content-only variant of the shared model, not the
  original MeLU architecture).
* `cuda` and `mps` are supported by the code but were not used for the reported
  numbers; the reference results were produced on CPU.
* Ties between a positive and a negative are resolved in favour of the positive
  (see `docs/PROTOCOL.md`). This is irrelevant for continuous cosine scores.

## Tests

```bash
make test
```

## Citation

To be completed after publication.

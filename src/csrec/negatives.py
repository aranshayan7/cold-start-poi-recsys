"""Fixed evaluation negatives, shared by every model.

For a given seed the negatives are generated once and cached on disk, so that
all learners and baselines are scored against exactly the same candidates.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Sequence, Tuple

import numpy as np

from .config import Config
from .data import Dataset, Task
from .geo import haversine


def sample_negatives(ds: Dataset, tasks: Sequence[Task], seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sample global and local negatives for each task.

    Global negatives are drawn uniformly from all unvisited POIs. Local negatives
    are drawn from the ``local_pool`` unvisited POIs nearest to the centroid of the
    user's support set, which gives a much harder, geographically plausible
    candidate set.
    """
    cfg = ds.cfg
    rng = np.random.RandomState(1000 + seed)
    n = cfg.q * cfg.neg_per_pos
    global_negs, local_negs = [], []
    for t in tasks:
        visited = ds.user_visited[t.u]

        cand: List[int] = []
        while len(cand) < n:
            draw = rng.randint(0, ds.n_venues, size=n * 2)
            cand.extend(x for x in draw if x not in visited)
        global_negs.append(np.array(list(dict.fromkeys(cand))[:n]))

        sup = np.array(t.sup)
        dist = haversine(ds.lat[sup].mean(), ds.lon[sup].mean(), ds.lat, ds.lon)
        order = np.argsort(dist)
        pool = [x for x in order[: cfg.local_pool + len(visited) + 10] if x not in visited][: cfg.local_pool]
        local_negs.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    return global_negs, local_negs


def load_or_build(ds: Dataset, cfg: Config):
    """Return ``(test_global, test_local, val_global)``, using an on-disk cache."""
    path = os.path.join(cfg.out, f"negs_k{cfg.k}_s{cfg.seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    test_g, test_l = sample_negatives(ds, ds.test_tasks, cfg.seed)
    val_g, _ = sample_negatives(ds, ds.val_subset, cfg.seed + 7)
    with open(path, "wb") as fh:
        pickle.dump((test_g, test_l, val_g), fh)
    return test_g, test_l, val_g

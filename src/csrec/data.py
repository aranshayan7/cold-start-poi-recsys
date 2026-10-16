"""Data loading and construction of per-user meta-learning tasks.

Expected files in the data directory
------------------------------------
checkins.parquet
    columns ``user_idx``, ``venue_idx``, ``utc_time`` (Foursquare time format).
pois.parquet
    columns ``venue_idx``, ``latitude``, ``longitude``, ``lat_norm``,
    ``lon_norm``, ``category_idx``.
splits.pkl
    dict with ``train_users``, ``val_users`` and ``test_users``. The splits are
    disjoint at the user level, which is what makes the setting user cold start.
social_adj.pkl
    dict mapping a user index to the list of friend indices.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config

TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass
class Task:
    """One user as a meta-learning task.

    ``sup`` is the support set (earliest unique POIs), ``qry`` the query set
    (the POIs visited right after), ``fr`` the support sets of up to
    ``n_friends`` training friends and ``pool`` an optional array of nearby
    unvisited POIs used for mixed training negatives.
    """

    u: int
    sup: List[int]
    qry: List[int]
    fr: List[List[int]] = field(default_factory=list)
    pool: np.ndarray | None = None


def unique_in_order(items) -> List[int]:
    """Deduplicate while keeping the first occurrence of each item."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Dataset:
    """In-memory view of the processed check-in data."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.data
        checkins = pd.read_parquet(os.path.join(d, "checkins.parquet"), columns=["user_idx", "venue_idx", "utc_time"])
        pois = pd.read_parquet(os.path.join(d, "pois.parquet")).sort_values("venue_idx")
        with open(os.path.join(d, "splits.pkl"), "rb") as fh:
            self.splits = pickle.load(fh)
        with open(os.path.join(d, "social_adj.pkl"), "rb") as fh:
            self.social: Dict[int, List[int]] = pickle.load(fh)

        self.n_venues = int(pois.venue_idx.max()) + 1
        self.n_categories = int(pois.category_idx.max()) + 1
        self._build_poi_tables(pois)

        # Chronological order is essential: the support set must precede the query set.
        checkins["t"] = pd.to_datetime(checkins.utc_time, format=TIME_FORMAT, errors="coerce")
        checkins = checkins.sort_values(["user_idx", "t"], kind="stable")
        self.user_seq = {u: unique_in_order(g.tolist()) for u, g in checkins.groupby("user_idx", sort=False)["venue_idx"]}
        self.user_visited = {u: set(s) for u, s in self.user_seq.items()}

        self.train_set = set(self.splits["train_users"])
        train_visits = checkins[checkins.user_idx.isin(self.train_set)].venue_idx.values
        self.popularity = np.bincount(train_visits, minlength=self.n_venues).astype(np.float64)
        self.seen_in_train = self.popularity > 0

        self.train_tasks = self.make_tasks(self.splits["train_users"])
        self.val_tasks = self.make_tasks(self.splits["val_users"])
        self.test_tasks = self.make_tasks(self.splits["test_users"])
        # A fixed random subset of validation users is used for early stopping (speed).
        order = np.random.RandomState(0).permutation(len(self.val_tasks))[: cfg.n_val]
        self.val_subset = [self.val_tasks[i] for i in order]

    def _build_poi_tables(self, pois: pd.DataFrame) -> None:
        """Dense content matrix [lat_norm, lon_norm, one-hot category] and raw coordinates."""
        idx = pois.venue_idx.values
        feat = np.zeros((self.n_venues, 2 + self.n_categories), dtype=np.float32)
        feat[idx, 0] = pois.lat_norm.values
        feat[idx, 1] = pois.lon_norm.values
        feat[idx, 2 + pois.category_idx.values.astype(int)] = 1.0
        self.features = feat
        self.lat = np.zeros(self.n_venues)
        self.lon = np.zeros(self.n_venues)
        self.lat[idx] = pois.latitude.values
        self.lon[idx] = pois.longitude.values

    def make_tasks(self, users) -> List[Task]:
        """Build tasks for users with at least ``k_max + q`` distinct POIs.

        The support set is the first ``k`` of the first ``k_max`` unique POIs and
        the query set is always the next ``q`` POIs, so changing ``k`` never
        changes what is being predicted.
        """
        c = self.cfg
        tasks = []
        for u in users:
            seq = self.user_seq.get(u, [])
            if len(seq) < c.k_max + c.q:
                continue
            friends = [
                f for f in self.social.get(u, [])
                if f in self.train_set and len(self.user_seq.get(f, [])) >= c.k_max
            ][: c.n_friends]
            tasks.append(
                Task(
                    u=u,
                    sup=seq[: c.k_max][: c.k],
                    qry=seq[c.k_max : c.k_max + c.q],
                    fr=[self.user_seq[f][: c.k_max] for f in friends],
                )
            )
        return tasks

    def summary(self) -> str:
        te = self.test_tasks
        with_friends = np.mean([len(t.fr) > 0 for t in te])
        unseen = np.mean([not self.seen_in_train[v] for t in te for v in t.qry])
        return (
            f"venues={self.n_venues} cats={self.n_categories} "
            f"tasks train/val/test={len(self.train_tasks)}/{len(self.val_tasks)}/{len(te)} "
            f"with_friends(test)={with_friends:.3f} query_unseen_in_train={unseen:.3f}"
        )

"""Non-learned reference scorers."""
from __future__ import annotations

import numpy as np

from .data import Dataset, Task
from .geo import haversine


def popularity_scorer(ds: Dataset, rng: np.random.RandomState):
    """Rank POIs by log training popularity (tiny noise breaks ties reproducibly)."""
    logpop = np.log1p(ds.popularity)

    def score(_: Task) -> np.ndarray:
        return logpop + 1e-6 * rng.rand(ds.n_venues)

    return score


def local_popularity_scorer(ds: Dataset, distance_scale_km: float = 2.0):
    """Log popularity discounted linearly by distance to the support centroid."""
    logpop = np.log1p(ds.popularity)

    def score(t: Task) -> np.ndarray:
        sup = np.array(t.sup)
        dist = haversine(ds.lat[sup].mean(), ds.lon[sup].mean(), ds.lat, ds.lon)
        return logpop - dist / distance_scale_km

    return score

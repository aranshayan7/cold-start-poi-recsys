"""Ranking metrics and the evaluation loop."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .data import Dataset, Task

CUTOFFS = (10, 20)


def ranking_metrics(score_pos: np.ndarray, score_neg: np.ndarray, cutoffs=CUTOFFS) -> Dict[str, float]:
    """NDCG@K and Recall@K for one user.

    The positives and negatives are ranked together with a stable sort. Because
    the positives are placed first in the concatenated array, an exact tie between
    a positive and a negative is resolved in favour of the positive. Scores are
    continuous cosine similarities, so ties are practically absent for the
    learned models; see ``docs/PROTOCOL.md`` for the discussion.
    """
    scores = np.concatenate([score_pos, score_neg])
    labels = np.concatenate([np.ones(len(score_pos)), np.zeros(len(score_neg))])
    labels = labels[np.argsort(-scores, kind="stable")]
    n_pos = len(score_pos)
    out = {}
    for k in cutoffs:
        top = labels[:k]
        dcg = (top / np.log2(np.arange(2, k + 2))).sum()
        ideal = (1 / np.log2(np.arange(2, min(n_pos, k) + 2))).sum()
        out[f"NDCG@{k}"] = dcg / ideal
        out[f"Recall@{k}"] = top.sum() / n_pos
    return out


def evaluate(
    ds: Dataset,
    score_fn: Callable[[Task], np.ndarray],
    tasks: Sequence[Task],
    global_negs: Sequence[np.ndarray],
    local_negs: Optional[Sequence[np.ndarray]] = None,
    full: bool = False,
) -> Dict[str, List[Dict[str, float]]]:
    """Per-user metrics under up to three candidate-set protocols.

    ``global``: uniform sampled negatives (protocol of the original thesis).
    ``local``: negatives sampled near the user's support centroid.
    ``full``: ranking against every unvisited POI (no sampling).
    """
    res = {"global": [], "local": [], "full": []}
    for i, t in enumerate(tasks):
        s = score_fn(t)
        pos = s[t.qry]
        res["global"].append(ranking_metrics(pos, s[global_negs[i]]))
        if local_negs is not None:
            res["local"].append(ranking_metrics(pos, s[local_negs[i]]))
        if full:
            mask = np.ones(ds.n_venues, bool)
            mask[list(ds.user_visited[t.u])] = False
            res["full"].append(ranking_metrics(pos, s[mask]))
    return {k: v for k, v in res.items() if v}


def summarise(res: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Average per-user metrics within each protocol."""
    return {setting: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} for setting, rows in res.items()}

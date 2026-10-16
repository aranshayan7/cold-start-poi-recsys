"""Unit tests for the evaluation code and a miniature end-to-end check."""
import numpy as np
import pytest

from csrec.config import Config
from csrec.data import Dataset, unique_in_order
from csrec.geo import haversine
from csrec.metrics import ranking_metrics


def test_perfect_ranking_has_unit_ndcg():
    pos = np.array([5.0, 4.0, 3.0])
    neg = np.arange(100, dtype=float) / 100.0
    m = ranking_metrics(pos, neg)
    assert m["NDCG@10"] == pytest.approx(1.0)
    assert m["Recall@10"] == pytest.approx(1.0)


def test_worst_ranking_has_zero_metrics():
    pos = np.array([-1.0, -2.0])
    neg = np.arange(100, dtype=float)
    m = ranking_metrics(pos, neg)
    assert m["NDCG@10"] == 0.0
    assert m["Recall@20"] == 0.0


def test_recall_counts_hits_in_top_k():
    pos = np.array([10.0, 0.5])            # one hit at rank 1, one buried
    neg = np.linspace(1.0, 2.0, 50)
    m = ranking_metrics(pos, neg)
    assert m["Recall@10"] == pytest.approx(0.5)


def test_haversine_known_distance():
    # Tokyo station to Osaka station is roughly 400 km.
    d = haversine(35.681, 139.767, 34.702, 135.495)
    assert 390 < d < 410
    assert haversine(35.0, 139.0, 35.0, 139.0) == pytest.approx(0.0)


def test_unique_in_order_keeps_first_occurrence():
    assert unique_in_order([3, 1, 3, 2, 1]) == [3, 1, 2]


@pytest.fixture(scope="module")
def dataset():
    return Dataset(Config(data="data/sample", n_val=20))


def test_tasks_are_chronological_and_disjoint(dataset):
    c = dataset.cfg
    for t in dataset.test_tasks[:50]:
        seq = dataset.user_seq[t.u]
        assert t.sup == seq[: c.k_max][: c.k]
        assert t.qry == seq[c.k_max : c.k_max + c.q]
        assert not set(t.sup) & set(t.qry)


def test_splits_are_user_disjoint(dataset):
    s = dataset.splits
    assert not set(s["train_users"]) & set(s["val_users"])
    assert not set(s["train_users"]) & set(s["test_users"])
    assert not set(s["val_users"]) & set(s["test_users"])


def test_friend_lists_are_capped_and_present(dataset):
    for t in dataset.test_tasks:
        assert len(t.fr) <= dataset.cfg.n_friends
    assert any(t.fr for t in dataset.test_tasks)

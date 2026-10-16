"""End-to-end test of the preprocessing on a tiny synthetic raw dataset."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_data", os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_data.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture()
def raw_dir(tmp_path):
    rng = np.random.RandomState(0)
    venues = [f"v{i:03d}" for i in range(40)]
    cats = ["Cafe", "Ramen", "Park", "Station", "Museum"]
    with open(tmp_path / "raw_POIs.txt", "w") as fh:
        for i, v in enumerate(venues):
            fh.write(f"{v}\t{35 + i * 0.01}\t{139 + i * 0.01}\t{cats[i % 5]}\tJP\n")
        fh.write("far\t10.0\t10.0\tCafe\tUS\n")          # other country
        fh.write("bad\t60.0\t139.0\tCafe\tJP\n")         # outside the bounding box
    t = "Fri Apr 05 00:00:{:02d} +0000 2013"
    with open(tmp_path / "dataset_WWW_Checkins_anonymized.txt", "w") as fh:
        for u in range(1, 31):
            n = 25 if u <= 28 else (5 if u == 29 else 500)   # user 29 too quiet, user 30 too active
            for j in range(n):
                fh.write(f"{u}\t{venues[rng.randint(0, 40)]}\t{t.format(j % 60)}\t540\n")
        fh.write("1\tv001\tgarbled\t540\textra-field\n")        # corrupted row
        fh.write("1\tv001\tnot a time\t540\n")                  # unparsable timestamp
        fh.write("1\tfar\t" + t.format(1) + "\t540\n")        # venue of another country
    with open(tmp_path / "dataset_WWW_friendship_new.txt", "w") as fh:
        for u in range(1, 28):
            fh.write(f"{u}\t{u + 1}\n")
        fh.write("1\t999\n")                                    # friend that does not exist
    return str(tmp_path)


def test_pipeline_filters_indexes_and_splits(raw_dir):
    ck, pois, splits, social, stats = prepare.build(
        raw_dir, ["JP"], min_user_checkins=20, max_user_checkins=100, min_venue_freq=5, top_categories=3, seed=42)

    assert stats["users"] == 28                                   # quiet and over-active users removed
    assert set(ck.user_idx) == set(range(28))                     # dense user indices
    assert list(pois.venue_idx) == list(range(len(pois)))         # dense, ordered venue indices
    assert list(pois.venue_id) == sorted(pois.venue_id)           # indices follow sorted identifiers
    assert "far" not in set(pois.venue_id) and "bad" not in set(pois.venue_id)
    assert pois.category_encoded.nunique() <= 4                   # 3 kept categories + "Other"
    assert abs(pois.lat_norm.mean()) < 1e-9

    everyone = splits["train_users"] + splits["val_users"] + splits["test_users"]
    assert sorted(everyone) == list(range(28))                    # disjoint and exhaustive
    assert len(splits["train_users"]) > len(splits["val_users"]) >= len(splits["test_users"]) - 1

    assert all(f != u for u, fs in social.items() for f in fs)    # no self loops
    assert all(u in social[f] for u, fs in social.items() for f in fs)  # symmetric
    assert 999 not in social                                      # unknown friend ignored


def test_split_is_deterministic(raw_dir):
    a = prepare.build(raw_dir, ["JP"], 20, 100, 5, 3, 42)[2]
    b = prepare.build(raw_dir, ["JP"], 20, 100, 5, 3, 42)[2]
    assert a == b

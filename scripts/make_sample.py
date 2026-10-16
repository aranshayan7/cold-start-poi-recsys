#!/usr/bin/env python3
"""Create a small, self-contained sample of the processed data.

The sample keeps the user-level train/validation/test separation, the friendship
edges among the selected users and the first ``--max_checkins`` check-ins of each
user. POIs are re-indexed so the sample is a valid input on its own. It is meant
for smoke tests and for getting familiar with the pipeline in minutes, not for
reporting results.

Example
-------
    python scripts/make_sample.py --src /path/to/processed_japan_v2 --dst data/sample
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory with the full processed files")
    ap.add_argument("--dst", default="data/sample")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--min_unique", type=int, default=30, help="minimum distinct POIs per user (k_max + q)")
    ap.add_argument("--max_checkins", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.RandomState(a.seed)
    ck = pd.read_parquet(os.path.join(a.src, "checkins.parquet"), columns=["user_idx", "venue_idx", "utc_time"])
    pois = pd.read_parquet(os.path.join(a.src, "pois.parquet"))
    splits = pickle.load(open(os.path.join(a.src, "splits.pkl"), "rb"))
    social = pickle.load(open(os.path.join(a.src, "social_adj.pkl"), "rb"))

    ck["t"] = pd.to_datetime(ck.utc_time, format=TIME_FORMAT, errors="coerce")
    ck = ck.sort_values(["user_idx", "t"], kind="stable")
    enough = set(ck.groupby("user_idx").venue_idx.nunique().loc[lambda s: s >= a.min_unique].index)

    chosen = {}
    for name, n in (("train_users", a.n_train), ("val_users", a.n_val), ("test_users", a.n_test)):
        pool = sorted(u for u in splits[name] if u in enough)
        chosen[name] = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    keep = set().union(*chosen.values())

    # Re-index users and POIs so that indices are dense.
    sub = ck[ck.user_idx.isin(keep)].groupby("user_idx", sort=False).head(a.max_checkins)
    users = {u: i for i, u in enumerate(sorted(keep))}
    venues = {v: i for i, v in enumerate(sorted(sub.venue_idx.unique()))}
    out_ck = pd.DataFrame({
        "user_idx": sub.user_idx.map(users).astype("int64"),
        "venue_idx": sub.venue_idx.map(venues).astype("int64"),
        "utc_time": sub.utc_time.values,
    })
    out_pois = pois[pois.venue_idx.isin(venues)].copy()
    out_pois["venue_idx"] = out_pois.venue_idx.map(venues).astype("int64")
    out_pois = out_pois[["venue_idx", "latitude", "longitude", "lat_norm", "lon_norm", "category_idx"]]
    out_pois = out_pois.sort_values("venue_idx").reset_index(drop=True)
    out_splits = {k: [users[u] for u in v] for k, v in chosen.items()}
    out_social = {users[u]: [users[f] for f in social.get(u, []) if f in users] for u in keep}

    os.makedirs(a.dst, exist_ok=True)
    out_ck.to_parquet(os.path.join(a.dst, "checkins.parquet"), index=False)
    out_pois.to_parquet(os.path.join(a.dst, "pois.parquet"), index=False)
    pickle.dump(out_splits, open(os.path.join(a.dst, "splits.pkl"), "wb"))
    pickle.dump(out_social, open(os.path.join(a.dst, "social_adj.pkl"), "wb"))
    deg = np.mean([len(v) for v in out_social.values()])
    print(f"users={len(users)} venues={len(venues)} checkins={len(out_ck)} mean_friends={deg:.1f}")
    print({k: len(v) for k, v in out_splits.items()})


if __name__ == "__main__":
    main()

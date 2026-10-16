#!/usr/bin/env python3
"""Compare two processed data directories (for example a fresh preparation against the
files used for the reported results) and print the agreement per component.

    python scripts/compare_processed.py --a data/processed --b /path/to/reference
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd


def load(d):
    ck = pd.read_parquet(os.path.join(d, "checkins.parquet"))
    po = pd.read_parquet(os.path.join(d, "pois.parquet"))
    sp = pickle.load(open(os.path.join(d, "splits.pkl"), "rb"))
    so = pickle.load(open(os.path.join(d, "social_adj.pkl"), "rb"))
    return ck, po, sp, so


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    (ca, pa, sa, na), (cb, pb, sb, nb) = load(args.a), load(args.b)

    print(f"users      a={ca.user_id.nunique():,}  b={cb.user_id.nunique():,}  "
          f"identical set={set(ca.user_id) == set(cb.user_id)}")
    va, vb = set(pa.venue_id), set(pb.venue_id)
    print(f"venues     a={len(va):,}  b={len(vb):,}  shared={len(va & vb):,}  only_a={len(va - vb)}  only_b={len(vb - va)}")
    print(f"check-ins  a={len(ca):,}  b={len(cb):,}  difference={len(ca) - len(cb):+,}")

    # Index agreement: map each user/venue id to its index on both sides.
    ua = ca.drop_duplicates("user_id").set_index("user_id").user_idx
    ub = cb.drop_duplicates("user_id").set_index("user_id").user_idx
    common = ua.index.intersection(ub.index)
    print(f"user index identical for {np.mean(ua[common].values == ub[common].values):.4f} of shared users")

    m = pa.merge(pb, on="venue_id", suffixes=("_a", "_b"))
    print(f"category label identical for {np.mean(m.category_encoded_a == m.category_encoded_b):.4f} of shared POIs; "
          f"coordinates identical for {np.mean(np.isclose(m.latitude_a, m.latitude_b)):.4f}")

    # Split agreement, expressed on original user ids.
    ida = ca.drop_duplicates("user_idx").set_index("user_idx").user_id
    idb = cb.drop_duplicates("user_idx").set_index("user_idx").user_id
    for part in ("train_users", "val_users", "test_users"):
        s1, s2 = {ida[i] for i in sa[part]}, {idb[i] for i in sb[part]}
        print(f"{part:12s} a={len(s1):,} b={len(s2):,} overlap={len(s1 & s2) / max(len(s1 | s2), 1):.4f} (Jaccard)")

    print(f"social lists: a={len(na):,} users, b={len(nb):,} users, "
          f"edges a={sum(map(len, na.values())) // 2:,} b={sum(map(len, nb.values())) // 2:,}")


if __name__ == "__main__":
    main()

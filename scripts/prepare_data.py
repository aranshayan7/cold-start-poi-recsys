#!/usr/bin/env python3
"""Build the processed data files from the public Foursquare check-in dataset.

Input files (Yang et al., WWW 2019; place them in one directory):
    dataset_WWW_Checkins_anonymized.txt   user, venue, UTC time, timezone offset (tab separated)
    raw_POIs.txt                          venue, latitude, longitude, category, country (tab separated)
    dataset_WWW_friendship_new.txt        user, user (tab separated)

Output files (see data/README.md for the format):
    checkins.parquet, pois.parquet, splits.pkl, social_adj.pkl, stats.json

Processing steps
    1. Keep POIs of the requested countries inside the country bounding boxes.
    2. Keep check-ins at those POIs (streamed in chunks, so the 1.5 GB file fits in memory).
    3. Drop rows with an unparsable timestamp, venues visited fewer than ``--min_venue_freq``
       times, then users with fewer than ``--min_user_checkins`` or more than
       ``--max_user_checkins`` check-ins (the latter removes bots and automated accounts).
    4. Re-index users and venues densely, in increasing order of the original identifiers.
    5. Encode categories: the ``--top_categories`` most frequent categories (by number of
       POIs) are kept, all others become "Other". Integer codes follow first appearance.
    6. Standardise latitude and longitude (z-score over the retained POIs).
    7. Build the symmetric friendship lists among retained users.
    8. Split users 60/20/20 into train/validation/test, stratified by activity tercile
       (number of check-ins), with a fixed seed. The split is at user level, which makes
       every test user a cold-start user.

Examples
--------
    python scripts/prepare_data.py --raw data/raw --out data/processed                  # Japan
    python scripts/prepare_data.py --raw data/raw --out data/processed_asia \\
        --countries JP,KR,SG,TW,HK,CN
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# Bounding boxes (lat_min, lat_max, lon_min, lon_max) used to discard mislabelled POIs.
BOUNDING_BOXES = {
    "JP": (30.0, 45.0, 129.0, 146.0),
    "KR": (34.0, 38.5, 126.0, 130.0),
    "SG": (1.0, 1.5, 103.5, 104.1),
    "TW": (21.5, 25.5, 119.5, 122.1),
    "HK": (22.1, 22.6, 113.8, 114.5),
    "CN": (18.0, 53.5, 73.5, 135.0),
}
CHECKIN_FILE = "dataset_WWW_Checkins_anonymized.txt"
POI_FILE = "raw_POIs.txt"
FRIEND_FILE = "dataset_WWW_friendship_new.txt"
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def load_pois(path, countries, chunk_rows=1_000_000):
    """Stream the POI file (11 million rows) and keep POIs of the requested countries."""
    kept = []
    reader = pd.read_csv(path, sep="\t", header=None, names=["venue_id", "latitude", "longitude", "category", "country"],
                         on_bad_lines="skip", dtype={"venue_id": str, "country": str}, chunksize=chunk_rows)
    for chunk in reader:
        keep = pd.Series(False, index=chunk.index)
        for c in countries:
            mask = chunk.country == c
            if c in BOUNDING_BOXES:
                la0, la1, lo0, lo1 = BOUNDING_BOXES[c]
                mask &= chunk.latitude.between(la0, la1) & chunk.longitude.between(lo0, lo1)
            keep |= mask
        kept.append(chunk[keep])
    pois = pd.concat(kept, ignore_index=True).drop_duplicates("venue_id").reset_index(drop=True)
    for c in countries:
        print(f"  {c}: {int((pois.country == c).sum()):,} POIs")
    return pois


def load_checkins(path, venue_ids, chunk_rows=2_000_000):
    """Stream the check-in file and keep rows whose venue is in ``venue_ids``."""
    parts = []
    reader = pd.read_csv(path, sep="\t", header=None, names=["user_id", "venue_id", "utc_time", "timezone_offset"],
                         dtype={"user_id": np.int64, "venue_id": str, "utc_time": str, "timezone_offset": np.int64},
                         on_bad_lines="skip", chunksize=chunk_rows)  # the raw file has a few corrupted rows
    for i, chunk in enumerate(reader, 1):
        chunk = chunk[chunk.venue_id.isin(venue_ids)]
        valid = pd.to_datetime(chunk.utc_time, format=TIME_FORMAT, errors="coerce").notna()
        parts.append(chunk[valid])
        print(f"  chunk {i}: kept {sum(len(p) for p in parts):,} check-ins", flush=True)
    return pd.concat(parts, ignore_index=True)


def stratified_split(counts, seed):
    """60/20/20 user split, stratified by activity tercile; ``counts`` is indexed by user_idx."""
    q33, q66 = counts.quantile(0.33), counts.quantile(0.66)
    n = len(counts)
    users = np.arange(n)
    c = counts.reindex(users).fillna(0).values
    groups = [users[c <= q33], users[(c > q33) & (c <= q66)], users[c > q66]]
    np.random.seed(seed)
    train, val, test = [], [], []
    for g in groups:
        g = g.copy()
        np.random.shuffle(g)
        m = len(g)
        train += g[: int(m * 0.6)].tolist()
        val += g[int(m * 0.6): int(m * 0.8)].tolist()
        test += g[int(m * 0.8):].tolist()
    return train, val, test


def build(raw_dir, countries, min_user_checkins, max_user_checkins, min_venue_freq, top_categories, seed):
    print("[1/6] POIs")
    pois = load_pois(os.path.join(raw_dir, POI_FILE), countries)
    print(f"  total: {len(pois):,}")

    print("[2/6] Check-ins")
    ck = load_checkins(os.path.join(raw_dir, CHECKIN_FILE), set(pois.venue_id))

    print("[3/6] Filtering")
    freq = ck.venue_id.value_counts()
    ck = ck[ck.venue_id.isin(freq[freq >= min_venue_freq].index)]
    per_user = ck.user_id.value_counts()
    keep_users = per_user[(per_user >= min_user_checkins) & (per_user <= max_user_checkins)].index
    ck = ck[ck.user_id.isin(keep_users)].copy()
    pois = pois[pois.venue_id.isin(set(ck.venue_id))].copy()
    print(f"  users={ck.user_id.nunique():,} venues={len(pois):,} check-ins={len(ck):,}")

    print("[4/6] Indexing and encoding")
    user_ids = np.sort(ck.user_id.unique())
    venue_ids = np.sort(pois.venue_id.unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    venue_idx = {v: i for i, v in enumerate(venue_ids)}
    ck["user_idx"] = ck.user_id.map(user_idx).astype("int64")
    ck["venue_idx"] = ck.venue_id.map(venue_idx).astype("int64")
    pois["venue_idx"] = pois.venue_id.map(venue_idx).astype("int64")
    pois = pois.sort_values("venue_idx").reset_index(drop=True)

    top = set(pois.category.value_counts().head(top_categories).index)
    pois["category_encoded"] = np.where(pois.category.isin(top), pois.category, "Other")
    pois["category_idx"] = pd.factorize(pois.category_encoded)[0].astype("int64")
    for col, norm in (("latitude", "lat_norm"), ("longitude", "lon_norm")):
        pois[norm] = (pois[col] - pois[col].mean()) / (pois[col].std() + 1e-8)

    print("[5/6] Social graph")
    friends = pd.read_csv(os.path.join(raw_dir, FRIEND_FILE), sep="\t", header=None, names=["u", "v"])
    friends = friends[friends.u.isin(user_idx) & friends.v.isin(user_idx)]
    social = defaultdict(list)
    for a, b in zip(friends.u.map(user_idx).values, friends.v.map(user_idx).values):
        social[int(a)].append(int(b))
        social[int(b)].append(int(a))
    social = {u: sorted(set(v)) for u, v in social.items()}

    print("[6/6] Split")
    counts = ck.user_idx.value_counts()
    train, val, test = stratified_split(counts, seed)
    splits = {"train_users": train, "val_users": val, "test_users": test, "setting": "inductive"}

    stats = {
        "countries": countries, "users": int(len(user_ids)), "venues": int(len(pois)), "checkins": int(len(ck)),
        "categories": int(pois.category_idx.nunique()), "friendship_edges": int(sum(len(v) for v in social.values()) // 2),
        "train_val_test": [len(train), len(val), len(test)],
        "min_user_checkins": min_user_checkins, "max_user_checkins": max_user_checkins, "min_venue_freq": min_venue_freq, "seed": seed,
    }
    ck_out = ck[["user_id", "venue_id", "utc_time", "timezone_offset", "user_idx", "venue_idx"]]
    pois_out = pois[["venue_id", "latitude", "longitude", "category", "country", "venue_idx", "lat_norm", "lon_norm",
                     "category_encoded", "category_idx"]]
    return ck_out, pois_out, splits, social, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="directory with the three raw files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", default="JP", help="comma separated ISO country codes")
    ap.add_argument("--min_user_checkins", type=int, default=20)
    ap.add_argument("--max_user_checkins", type=int, default=1000)
    ap.add_argument("--min_venue_freq", type=int, default=5)
    ap.add_argument("--top_categories", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    countries = [c.strip().upper() for c in a.countries.split(",") if c.strip()]
    ck, pois, splits, social, stats = build(a.raw, countries, a.min_user_checkins,
                                            a.max_user_checkins, a.min_venue_freq, a.top_categories, a.seed)
    os.makedirs(a.out, exist_ok=True)
    ck.to_parquet(os.path.join(a.out, "checkins.parquet"), index=False)
    pois.to_parquet(os.path.join(a.out, "pois.parquet"), index=False)
    pickle.dump(splits, open(os.path.join(a.out, "splits.pkl"), "wb"))
    pickle.dump(social, open(os.path.join(a.out, "social_adj.pkl"), "wb"))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

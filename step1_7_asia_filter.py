"""
Step 1.7: Filter to Asia Subset
=============================================================
Countries: JP, KR, SG, TW, HK, CN
Same params as Japan v2:
  - min_user_checkins: 20
  - min_venue_freq: 5
  - Bounding box per country
  - Stratified split
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import pathlib

# _BASE      = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet")

SOURCE_DIR = str(_BASE / "processed")
OUTPUT_DIR = str(_BASE / "processed_asia_v1")
DATA_DIR   = str(_BASE / "data")

MIN_USER_CHECKINS = 20
MIN_VENUE_FREQ    = 5

# Bounding boxes per country
ASIA_COUNTRIES = {
    'JP': {'lat': (30.0, 45.0),  'lon': (129.0, 146.0)},
    'KR': {'lat': (34.0, 38.5),  'lon': (126.0, 130.0)},
    'SG': {'lat': (1.0,  1.5),   'lon': (103.5, 104.1)},
    'TW': {'lat': (21.5, 25.5),  'lon': (119.5, 122.1)},
    'HK': {'lat': (22.1, 22.6),  'lon': (113.8, 114.5)},
    'CN': {'lat': (18.0, 53.5),  'lon': (73.5,  135.0)},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1.7: Filtering to Asia Subset")
print(f"Countries: {list(ASIA_COUNTRIES.keys())}")
print("=" * 60)

# ===================== 1. LOAD RAW POIS =====================
print("\n[1/6] Loading POIs...")
poi_raw = pd.read_csv(
    os.path.join(DATA_DIR, "raw_POIs.txt"),
    sep='\t', header=None,
    names=['venue_id', 'lat', 'lon', 'category', 'country'],
    on_bad_lines='skip'
)

# Filter by country + bounding box
asia_masks = []
for country, bbox in ASIA_COUNTRIES.items():
    mask = (
        (poi_raw['country'] == country) &
        (poi_raw['lat'] >= bbox['lat'][0]) &
        (poi_raw['lat'] <= bbox['lat'][1]) &
        (poi_raw['lon'] >= bbox['lon'][0]) &
        (poi_raw['lon'] <= bbox['lon'][1])
    )
    n = mask.sum()
    print(f"  {country}: {n:,} POIs")
    asia_masks.append(mask)

combined_mask = asia_masks[0]
for m in asia_masks[1:]:
    combined_mask = combined_mask | m

asia_pois_raw = poi_raw[combined_mask]
asia_venue_ids = set(asia_pois_raw['venue_id'].values)
print(f"  Total Asia POIs: {len(asia_venue_ids):,}")

# ===================== 2. LOAD PROCESSED DATA =====================
print("\n[2/6] Loading full processed data...")
checkins    = pd.read_parquet(os.path.join(SOURCE_DIR, "checkins.parquet"))
pois        = pd.read_parquet(os.path.join(SOURCE_DIR, "pois.parquet"))
friendships = pd.read_parquet(os.path.join(SOURCE_DIR, "friendships.parquet"))
with open(os.path.join(SOURCE_DIR, "mappings.pkl"), 'rb') as f:
    mappings = pickle.load(f)
print(f"  Loaded: {len(checkins):,} check-ins, {len(pois):,} POIs")

# ===================== 3. FILTER =====================
print("\n[3/6] Filtering...")
venue2id = mappings['venue2id']
asia_venue_idxs = {venue2id[v] for v in asia_venue_ids if v in venue2id}
print(f"  Asia venues in processed data: {len(asia_venue_idxs):,}")

asia_checkins = checkins[checkins['venue_idx'].isin(asia_venue_idxs)].copy()
print(f"  Asia check-ins (raw): {len(asia_checkins):,}")

# Popularity floor
venue_freq = asia_checkins['venue_idx'].value_counts()
popular_venues = set(venue_freq[venue_freq >= MIN_VENUE_FREQ].index)
asia_checkins = asia_checkins[asia_checkins['venue_idx'].isin(popular_venues)]
print(f"  After venue freq>={MIN_VENUE_FREQ}: {len(asia_checkins):,} check-ins, {len(popular_venues):,} venues")

# User threshold
user_counts = asia_checkins['user_idx'].value_counts()
active_users = set(user_counts[user_counts >= MIN_USER_CHECKINS].index)
asia_checkins = asia_checkins[asia_checkins['user_idx'].isin(active_users)]
print(f"  After user>={MIN_USER_CHECKINS}: {len(active_users):,} users")
print(f"  Final check-ins: {len(asia_checkins):,}")

# ===================== 4. REMAP IDS =====================
print("\n[4/6] Remapping IDs...")
unique_users  = sorted(asia_checkins['user_idx'].unique())
unique_venues = sorted(asia_checkins['venue_idx'].unique())

old_user_to_new  = {old: new for new, old in enumerate(unique_users)}
old_venue_to_new = {old: new for new, old in enumerate(unique_venues)}

asia_checkins = asia_checkins.copy()
asia_checkins['user_idx']  = asia_checkins['user_idx'].map(old_user_to_new)
asia_checkins['venue_idx'] = asia_checkins['venue_idx'].map(old_venue_to_new)

asia_pois = pois[pois['venue_idx'].isin(old_venue_to_new.keys())].copy()
asia_pois['venue_idx'] = asia_pois['venue_idx'].map(old_venue_to_new)

# Friendships
japan_user_set = set(old_user_to_new.keys())
asia_friends = friendships[
    friendships['user1_idx'].isin(japan_user_set) &
    friendships['user2_idx'].isin(japan_user_set)
].copy()
asia_friends['user1_idx'] = asia_friends['user1_idx'].map(old_user_to_new)
asia_friends['user2_idx'] = asia_friends['user2_idx'].map(old_user_to_new)

# Social adj
social_adj = defaultdict(list)
for _, row in asia_friends.iterrows():
    u1, u2 = int(row['user1_idx']), int(row['user2_idx'])
    social_adj[u1].append(u2)
    social_adj[u2].append(u1)

# ===================== 5. STATS =====================
print("\n[5/6] Stats...")
n_users  = len(unique_users)
n_venues = len(unique_venues)

user_checkin_counts  = asia_checkins['user_idx'].value_counts()
venue_checkin_counts = asia_checkins['venue_idx'].value_counts()

def gini(arr):
    arr = np.sort(arr.values)
    n = len(arr)
    if n == 0 or arr.sum() == 0: return 0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n

print(f"  Users:     {n_users:,}")
print(f"  Venues:    {n_venues:,}")
print(f"  Check-ins: {len(asia_checkins):,}")
print(f"  Avg check-ins/user: {user_checkin_counts.mean():.1f}")
print(f"  Avg venues/user:    {asia_checkins.groupby('user_idx')['venue_idx'].nunique().mean():.1f}")
print(f"  Gini: {gini(venue_checkin_counts):.3f}")
print(f"  Friendships: {len(asia_friends):,}")

# ===================== 6. STRATIFIED SPLIT =====================
print("\n[6/6] Stratified split...")
q33 = user_checkin_counts.quantile(0.33)
q66 = user_checkin_counts.quantile(0.66)

low_users  = [u for u in range(n_users) if user_checkin_counts.get(u, 0) <= q33]
med_users  = [u for u in range(n_users) if q33 < user_checkin_counts.get(u, 0) <= q66]
high_users = [u for u in range(n_users) if user_checkin_counts.get(u, 0) > q66]

np.random.seed(42)
train_users, val_users, test_users = [], [], []
for group in [low_users, med_users, high_users]:
    arr = np.array(group)
    np.random.shuffle(arr)
    n = len(arr)
    train_users.extend(arr[:int(n * 0.6)].tolist())
    val_users.extend(arr[int(n * 0.6):int(n * 0.8)].tolist())
    test_users.extend(arr[int(n * 0.8):].tolist())

print(f"  Train/Val/Test: {len(train_users):,}/{len(val_users):,}/{len(test_users):,}")

# Category remap
new_cats = sorted(asia_pois['category_idx'].unique()) if 'category_idx' in asia_pois.columns else []
cat_remap = {old: new for new, old in enumerate(new_cats)}
if 'category_idx' in asia_pois.columns:
    asia_pois['category_idx'] = asia_pois['category_idx'].map(cat_remap)

# lat/lon norm
if 'latitude' in asia_pois.columns:
    asia_pois['lat_norm'] = ((asia_pois['latitude'] - asia_pois['latitude'].mean())
                             / (asia_pois['latitude'].std() + 1e-8))
    asia_pois['lon_norm'] = ((asia_pois['longitude'] - asia_pois['longitude'].mean())
                             / (asia_pois['longitude'].std() + 1e-8))

# ===================== SAVE =====================
print("\nSaving...")
asia_checkins.to_parquet(os.path.join(OUTPUT_DIR, "checkins.parquet"), index=False)
asia_pois.to_parquet(os.path.join(OUTPUT_DIR, "pois.parquet"), index=False)
asia_friends.to_parquet(os.path.join(OUTPUT_DIR, "friendships.parquet"), index=False)

with open(os.path.join(OUTPUT_DIR, "mappings.pkl"), 'wb') as f:
    pickle.dump({
        'user2id':  {f"u_{i}": i for i in range(n_users)},
        'venue2id': {f"v_{i}": i for i in range(n_venues)},
        'cat2id':   {f"cat_{i}": i for i in range(len(new_cats))},
        'old_user_to_new':  old_user_to_new,
        'old_venue_to_new': old_venue_to_new,
    }, f)

with open(os.path.join(OUTPUT_DIR, "splits.pkl"), 'wb') as f:
    pickle.dump({
        'train_users': train_users,
        'val_users':   val_users,
        'test_users':  test_users,
        'setting': 'inductive',
    }, f)

with open(os.path.join(OUTPUT_DIR, "social_adj.pkl"), 'wb') as f:
    pickle.dump(dict(social_adj), f)

print(f"\n✅ Asia v1 saved to: {OUTPUT_DIR}/")
print("=" * 60)
print("FINAL STATS:")
print(f"  Users:     {n_users:,}")
print(f"  Venues:    {n_venues:,}")
print(f"  Check-ins: {len(asia_checkins):,}")
print(f"  Split:     {len(train_users):,}/{len(val_users):,}/{len(test_users):,}")
print(f"  (Japan was: 6,817 users, 66,617 venues)")
print("=" * 60)

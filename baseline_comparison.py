"""
Baseline Comparison - Fixed Version
=============================================================
Bug fix: query venues باید در candidate pool باشن
(فقط support باید excluded باشه، نه query)
=============================================================
"""

import os
import torch
import numpy as np
import pickle
import json
from datetime import datetime
from collections import defaultdict

import pathlib
# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Volumes/Extreme SSD/Model/Tourism_ProtoNet")
# PROCESSED_DIR = str(_BASE / "processed_japan_v2")
# PROCESSED_DIR = str(_BASE / "processed_asia_v1")
PROCESSED_DIR = str(_BASE / "processed")
RESULTS_DIR   = str(_BASE / "ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.append(str(_BASE))

from src.step2_dataset import load_processed_data

# ===================== LOAD DATA =====================
print("Loading Full v2 data...")
checkins, pois, mappings, splits, social_adj = load_processed_data(PROCESSED_DIR)

n_venues = len(mappings['venue2id'])
val_users = splits['val_users']
print(f"  n_venues={n_venues:,} | val_users={len(val_users):,}")

# ===================== LOOKUP TABLES =====================
print("Building lookup tables...")
user_venues = checkins.groupby('user_idx')['venue_idx'].apply(list).to_dict()
venue_counts = checkins['venue_idx'].value_counts()
popular_venues = venue_counts.index.tolist()
print(f"  Top venue visits: {venue_counts.iloc[0]:,}")


def compute_ndcg(ranked_list, true_positives, k):
    dcg = sum(1.0 / np.log2(i + 2)
              for i, item in enumerate(ranked_list[:k])
              if item in true_positives)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(true_positives), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ===================== EVAL (Fixed) =====================
def evaluate_baseline(recommend_fn, val_users, user_venues, social_adj,
                      n_venues, n_neg=100, support_k=15, top_k_list=[10, 20],
                      max_users=500):
    metrics = defaultdict(list)
    rng = np.random.RandomState(42)
    users_evaluated = 0

    for uid in val_users:
        if users_evaluated >= max_users:
            break

        venues = user_venues.get(uid, [])
        if len(venues) < support_k + 1:
            continue

        rng_local = np.random.RandomState(uid)
        venues_copy = venues.copy()
        rng_local.shuffle(venues_copy)

        support = set(venues_copy[:support_k])
        query = set(venues_copy[support_k:support_k + 15]) - support
        if not query:
            continue

        # Fix: baseline فقط support رو exclude می‌کنه، نه query رو
        # یعنی query venues ممکنه توی recommendations باشن → hits ممکنه > 0 بشه
        recommended = recommend_fn(
            uid, support, social_adj, user_venues,
            popular_venues, n_venues, rng, top_k_list[-1]
        )

        for k in top_k_list:
            top_k_rec = recommended[:k]
            hits = len(set(top_k_rec) & query)
            metrics[f'Precision@{k}'].append(hits / k)
            metrics[f'Recall@{k}'].append(hits / len(query))
            metrics[f'NDCG@{k}'].append(compute_ndcg(top_k_rec, query, k))

        users_evaluated += 1

    return {k: np.mean(v) for k, v in metrics.items()}, users_evaluated


# ===================== BASELINES =====================
def random_recommend(uid, support, social_adj, user_venues,
                     popular_venues, n_venues, rng, top_k):
    """Random venues - فقط support exclude"""
    candidates = []
    attempts = 0
    while len(candidates) < top_k and attempts < top_k * 20:
        v = rng.randint(0, n_venues)
        if v not in support and v not in candidates:
            candidates.append(v)
        attempts += 1
    return candidates


def popularity_recommend(uid, support, social_adj, user_venues,
                         popular_venues, n_venues, rng, top_k):
    """Popular venues globally - فقط support exclude"""
    candidates = []
    for v in popular_venues:
        if v not in support:
            candidates.append(v)
        if len(candidates) >= top_k:
            break
    return candidates


def social_pop_recommend(uid, support, social_adj, user_venues,
                         popular_venues, n_venues, rng, top_k):
    """Popular venues among friends - فقط support exclude"""
    friends = social_adj.get(uid, [])

    if not friends:
        return popularity_recommend(uid, support, social_adj, user_venues,
                                    popular_venues, n_venues, rng, top_k)

    friend_counts = defaultdict(int)
    for fid in friends:
        for v in user_venues.get(fid, []):
            if v not in support:
                friend_counts[v] += 1

    if not friend_counts:
        return popularity_recommend(uid, support, social_adj, user_venues,
                                    popular_venues, n_venues, rng, top_k)

    sorted_v = sorted(friend_counts.items(), key=lambda x: x[1], reverse=True)
    return [v for v, _ in sorted_v[:top_k]]


# ===================== RUN =====================
baselines = [
    ('Random',            random_recommend),
    ('Popularity',        popularity_recommend),
    ('Social-Popularity', social_pop_recommend),
]

print("\nRunning baselines (500 val users)...")
print("=" * 75)

all_results = []
for name, fn in baselines:
    print(f"\n  [{name}]")
    metrics, n_eval = evaluate_baseline(
        fn, val_users, user_venues, social_adj,
        n_venues, n_neg=100, support_k=15,
        top_k_list=[10, 20], max_users=500
    )
    print(f"  Users: {n_eval} | NDCG@10={metrics['NDCG@10']:.4f} | "
          f"NDCG@20={metrics['NDCG@20']:.4f} | Recall@10={metrics['Recall@10']:.4f}")
    all_results.append({'name': name, 'n_eval': n_eval, 'metrics': metrics})

# ===================== TABLE =====================
print("\n" + "=" * 80)
print("BASELINE COMPARISON - FINAL TABLE")
print("=" * 80)
print(f"  {'Method':<25} {'NDCG@10':>10} {'NDCG@20':>10} {'Recall@10':>10} {'Recall@20':>10}")
print("-" * 80)

for r in all_results:
    m = r['metrics']
    print(f"  {r['name']:<25} {m['NDCG@10']:>10.4f} {m['NDCG@20']:>10.4f} "
          f"{m['Recall@10']:>10.4f} {m['Recall@20']:>10.4f}")

print("-" * 80)
our_ndcg = 0.3298
best_baseline = max(r['metrics']['NDCG@10'] for r in all_results)
improvement = (our_ndcg - best_baseline) / best_baseline * 100 if best_baseline > 0 else 0

print(f"  {'Our Model (ProtoNet)':<25} {'0.3298':>10} {'0.3597':>10} {'0.2729':>10} {'0.3622':>10}  ← BEST")
print("=" * 80)
print(f"\n  Our model vs best baseline: +{improvement:.1f}% improvement in NDCG@10")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(RESULTS_DIR, f"baseline_comparison_{ts}.json")
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"  Results saved to: {out_path}")
"""
Test Set Evaluation - Final Results
=============================================================
Best model رو load می‌کنه و روی test users می‌سنجه
این عدد نهایی هست که در پایان‌نامه گزارش می‌دی
=============================================================
"""

import os
import torch
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

import pathlib
# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet")

# PROCESSED_DIR  = str(_BASE / "processed_japan_v2")
# CHECKPOINT_DIR = str(_BASE / "checkpoints_protonet")
# PROCESSED_DIR  = str(_BASE / "processed_asia_v1")
# CHECKPOINT_DIR = str(_BASE / "checkpoints_asia")
PROCESSED_DIR  = str(pathlib.Path(r".\processed"))
CHECKPOINT_DIR = str(_BASE / "checkpoints_full")

RESULTS_DIR    = str(_BASE / "ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.append(str(_BASE))

from src.step2_dataset import (
    TourismMetaDataset, collate_meta_tasks,
    load_processed_data, build_poi_feature_matrix
)
from src.step3_model import ProtoNetGNN
from src.utils import compute_metrics

# ===================== LOAD DATA =====================
print("=" * 60)
print("TEST SET EVALUATION - Final Results")
print("=" * 60)

print("\nLoading data...")
checkins, pois, mappings, splits, social_adj = load_processed_data(PROCESSED_DIR)

n_venues = len(mappings['venue2id'])
n_cats = len(mappings.get('cat2id', {}))
poi_feature_dim = 2 + n_cats + 1

poi_features_np = build_poi_feature_matrix(pois, n_venues, mappings)
poi_features_tensor = torch.tensor(poi_features_np, dtype=torch.float32)

print(f"  n_venues={n_venues:,} | test_users={len(splits['test_users']):,}")

# ===================== LOAD MODEL =====================
print("\nLoading best model...")
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    print("Make sure you ran main.py first!")
    exit(1)

ckpt = torch.load(checkpoint_path, map_location='cpu')
print(f"  Loaded from epoch {ckpt['epoch']} | Val NDCG@10={ckpt['ndcg']:.4f}")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Device: {device}")

model = ProtoNetGNN(
    n_venues=n_venues,
    poi_feature_dim=poi_feature_dim,
    embed_dim=64,
    dropout=0.5,
    max_vocab_size=100_000,
    temperature=0.05,
    poi_features_tensor=poi_features_tensor
).to(device)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===================== TEST DATASET =====================
print("\nBuilding test dataset...")
test_ds = TourismMetaDataset(
    splits['test_users'], checkins, social_adj,
    support_k=15, query_size=15, n_venues=n_venues,
    cache_path=os.path.join(PROCESSED_DIR, 'cache_protonet_test.pkl'),
    pois_df=pois, shuffle_support=False
)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=4, shuffle=False,
    collate_fn=collate_meta_tasks, num_workers=0
)
print(f"  Test batches: {len(test_loader):,}")

# ===================== EVALUATE =====================
print("\nEvaluating on test set (100 negatives per positive)...")

all_metrics = defaultdict(list)
with torch.no_grad():
    for idx, tasks_batch in enumerate(test_loader):
        metrics = compute_metrics(model, tasks_batch, device, [10, 20])
        for k, v in metrics.items():
            all_metrics[k].append(v)

        if (idx + 1) % 50 == 0:
            print(f"  Batch [{idx+1}/{len(test_loader)}] | "
                  f"NDCG@10={np.mean(all_metrics['NDCG@10']):.4f}")

test_metrics = {k: np.mean(vs) for k, vs in all_metrics.items()}

# ===================== RESULTS =====================
print("\n" + "=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)
print(f"  NDCG@10:    {test_metrics['NDCG@10']:.4f}")
print(f"  NDCG@20:    {test_metrics['NDCG@20']:.4f}")
print(f"  Recall@10:  {test_metrics['Recall@10']:.4f}")
print(f"  Recall@20:  {test_metrics['Recall@20']:.4f}")
print(f"  Precision@10: {test_metrics['Precision@10']:.4f}")
print(f"  Precision@20: {test_metrics['Precision@20']:.4f}")

# Compare val vs test
print("\n" + "-" * 60)
print("  Comparison: Val vs Test")
print("-" * 60)
val_ndcg = ckpt['ndcg']
test_ndcg = test_metrics['NDCG@10']
diff = (test_ndcg - val_ndcg) / val_ndcg * 100
print(f"  Val  NDCG@10: {val_ndcg:.4f}")
print(f"  Test NDCG@10: {test_ndcg:.4f}  ({diff:+.1f}%)")

if abs(diff) < 5:
    print("  ✓ Val و Test نزدیک هستن → مدل generalize می‌کنه")
elif diff < -10:
    print("  ⚠ Test از Val کمتره → کمی overfit روی val")
else:
    print("  ✓ Test از Val بهتره → generalization خوبه")

print("=" * 60)

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = {
    'test_metrics': test_metrics,
    'val_ndcg': val_ndcg,
    'model_epoch': ckpt['epoch'],
    'timestamp': ts,
}
out_path = os.path.join(RESULTS_DIR, f"test_results_{ts}.json")
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to: {out_path}")

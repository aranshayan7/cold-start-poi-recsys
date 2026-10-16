# """
# Ablation Study - ProtoNet Clean Project
# =============================================================
# 4 variant رو اجرا می‌کنه و نتایج رو مقایسه می‌کنه:
#
# 1. Full Model         (social + POI features + hard negatives)
# 2. No Social          (بدون social neighbors)
# 3. No POI Features    (بدون lat/lon/category)
# 4. No Hard Negatives  (random negatives به جای hard)
#
# هر variant یه cache جدا داره تا با هم تداخل نداشته باشن
# =============================================================
# """
#
# import os
# import torch
# import numpy as np
# import time
# import json
# from datetime import datetime
# from collections import defaultdict
#
# # ===================== PATHS =====================
# import pathlib
# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
# PROCESSED_DIR = str(_BASE / "processed_japan_v2")
# RESULTS_DIR   = str(_BASE / "ablation_results")
# os.makedirs(RESULTS_DIR, exist_ok=True)
#
# import sys
# sys.path.append(str(_BASE))
#
# from src.step2_dataset import (
#     TourismMetaDataset, collate_meta_tasks,
#     load_processed_data, build_poi_feature_matrix
# )
# from src.step3_model import ProtoNetGNN
# from src.step4_train import train_one_epoch, EarlyStopping
# from src.utils import compute_metrics
#
# # ===================== BASE CONFIG =====================
# BASE_CONFIG = {
#     'embed_dim': 64,
#     'dropout': 0.5,
#     'max_vocab_size': 100_000,
#     'temperature': 0.05,
#     'support_k': 15,
#     'query_size': 15,
#     'lr': 0.001,
#     'weight_decay': 2e-4,
#     'clip_norm': 1.0,
#     'batch_size': 4,
#     'n_epochs': 15,
#     'patience': 5,
#     'eval_every': 1,
#     'top_k_list': [10, 20],
#     'seed': 42,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
# }
#
# # ===================== 4 VARIANTS =====================
# VARIANTS = [
#     {
#         'name': 'Full Model',
#         'tag': 'full',
#         'use_social': True,
#         'use_poi_features': True,
#         'use_hard_negatives': True,
#     },
#     {
#         'name': 'No Social',
#         'tag': 'no_social',
#         'use_social': False,
#         'use_poi_features': True,
#         'use_hard_negatives': True,
#     },
#     {
#         'name': 'No POI Features',
#         'tag': 'no_poi',
#         'use_social': True,
#         'use_poi_features': False,
#         'use_hard_negatives': True,
#     },
#     {
#         'name': 'No Hard Negatives',
#         'tag': 'no_hard_neg',
#         'use_social': True,
#         'use_poi_features': True,
#         'use_hard_negatives': False,
#     },
# ]
#
#
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#
#
# def evaluate_full(model, val_loader, device, config, max_batches=40):
#     model.eval()
#     all_metrics = defaultdict(list)
#     with torch.no_grad():
#         for idx, tasks_batch in enumerate(val_loader):
#             if idx >= max_batches:
#                 break
#             metrics = compute_metrics(model, tasks_batch, device, config['top_k_list'])
#             for k, v in metrics.items():
#                 all_metrics[k].append(v)
#     return {k: np.mean(vs) for k, vs in all_metrics.items()}
#
#
# def run_variant(variant, checkins, pois, mappings, splits, social_adj,
#                 poi_features_tensor, n_venues, poi_feature_dim, device):
#     """یک variant رو train و eval می‌کنه"""
#
#     print(f"\n{'='*60}")
#     print(f"Running: {variant['name']}")
#     print(f"{'='*60}")
#
#     set_seed(BASE_CONFIG['seed'])
#
#     cache_train = os.path.join(PROCESSED_DIR, f"cache_ablation_{variant['tag']}_train.pkl")
#     cache_val   = os.path.join(PROCESSED_DIR, f"cache_ablation_{variant['tag']}_val.pkl")
#
#     # اگه no_social باشه، social_adj خالی پاس می‌دیم
#     social_adj_use = social_adj if variant['use_social'] else {}
#
#     # اگه no_hard_neg باشه، pois_df پاس نمی‌دیم
#     pois_df_use = pois if variant['use_hard_negatives'] else None
#
#     train_ds = TourismMetaDataset(
#         splits['train_users'], checkins, social_adj_use,
#         BASE_CONFIG['support_k'], BASE_CONFIG['query_size'], n_venues,
#         cache_path=cache_train, shuffle_support=True,
#         pois_df=pois_df_use
#     )
#
#     val_ds = TourismMetaDataset(
#         splits['val_users'], checkins, social_adj_use,
#         BASE_CONFIG['support_k'], BASE_CONFIG['query_size'], n_venues,
#         cache_path=cache_val, shuffle_support=False,
#         pois_df=pois_df_use
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_ds, batch_size=BASE_CONFIG['batch_size'], shuffle=True,
#         collate_fn=collate_meta_tasks, num_workers=0, drop_last=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_ds, batch_size=BASE_CONFIG['batch_size'], shuffle=False,
#         collate_fn=collate_meta_tasks, num_workers=0
#     )
#
#     # اگه no_poi باشه، poi_features_tensor پاس نمی‌دیم
#     poi_feat_use = poi_features_tensor if variant['use_poi_features'] else None
#     # poi_feature_dim برای no_poi باید کوچیک‌تر باشه (فقط ID embedding)
#     # ولی چون مدل از combiner استفاده می‌کنه، None پاس می‌دیم
#     feat_dim_use = poi_feature_dim if variant['use_poi_features'] else poi_feature_dim
#
#     model = ProtoNetGNN(
#         n_venues=n_venues,
#         poi_feature_dim=feat_dim_use,
#         embed_dim=BASE_CONFIG['embed_dim'],
#         dropout=BASE_CONFIG['dropout'],
#         max_vocab_size=BASE_CONFIG['max_vocab_size'],
#         temperature=BASE_CONFIG['temperature'],
#         poi_features_tensor=poi_feat_use
#     ).to(device)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=BASE_CONFIG['lr'],
#         weight_decay=BASE_CONFIG['weight_decay']
#     )
#
#     early_stop = EarlyStopping(patience=BASE_CONFIG['patience'])
#     best_ndcg = -np.inf
#     best_metrics = {}
#     best_epoch = 0
#
#     history = []
#
#     for epoch in range(1, BASE_CONFIG['n_epochs'] + 1):
#         t0 = time.time()
#         train_loss = train_one_epoch(model, optimizer, train_loader, device, BASE_CONFIG)
#         val_metrics = evaluate_full(model, val_loader, device, BASE_CONFIG)
#         elapsed = time.time() - t0
#
#         cur_ndcg = val_metrics.get('NDCG@10', 0)
#         history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})
#
#         print(f"  Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
#               f"NDCG@10: {cur_ndcg:.4f} | NDCG@20: {val_metrics.get('NDCG@20', 0):.4f} | "
#               f"Time: {elapsed:.1f}s")
#
#         if cur_ndcg > best_ndcg:
#             best_ndcg = cur_ndcg
#             best_metrics = val_metrics.copy()
#             best_epoch = epoch
#
#         if early_stop.step(cur_ndcg, epoch):
#             break
#
#     print(f"\n  ✓ Best: epoch={best_epoch}, NDCG@10={best_ndcg:.4f}")
#
#     return {
#         'variant': variant['name'],
#         'best_epoch': best_epoch,
#         'best_metrics': best_metrics,
#         'history': history,
#     }
#
#
# def main():
#     print("=" * 60)
#     print("ABLATION STUDY - ProtoNet Tourism Cold-Start")
#     print("=" * 60)
#     print(f"Variants: {[v['name'] for v in VARIANTS]}")
#
#     device = torch.device(BASE_CONFIG['device'])
#     print(f"Device: {device}")
#
#     # Load data once
#     print("\nLoading Japan v2 data...")
#     checkins, pois, mappings, splits, social_adj = load_processed_data(PROCESSED_DIR)
#
#     n_venues = len(mappings['venue2id'])
#     n_cats = len(mappings.get('cat2id', {}))
#     poi_feature_dim = 2 + n_cats + 1
#
#     poi_features_np = build_poi_feature_matrix(pois, n_venues, mappings)
#     poi_features_tensor = torch.tensor(poi_features_np, dtype=torch.float32)
#
#     print(f"  n_venues={n_venues:,} | n_cats={n_cats} | poi_dim={poi_feature_dim}")
#
#     # Run all variants
#     all_results = []
#     for variant in VARIANTS:
#         result = run_variant(
#             variant, checkins, pois, mappings, splits, social_adj,
#             poi_features_tensor, n_venues, poi_feature_dim, device
#         )
#         all_results.append(result)
#
#     # ===================== PRINT SUMMARY TABLE =====================
#     print("\n" + "=" * 80)
#     print("ABLATION STUDY RESULTS")
#     print("=" * 80)
#     print(f"{'Variant':<25} {'NDCG@10':>10} {'NDCG@20':>10} {'Recall@10':>10} {'Recall@20':>10} {'Epoch':>6}")
#     print("-" * 80)
#
#     full_ndcg = None
#     for r in all_results:
#         m = r['best_metrics']
#         ndcg10 = m.get('NDCG@10', 0)
#         ndcg20 = m.get('NDCG@20', 0)
#         rec10  = m.get('Recall@10', 0)
#         rec20  = m.get('Recall@20', 0)
#         epoch  = r['best_epoch']
#
#         if r['variant'] == 'Full Model':
#             full_ndcg = ndcg10
#
#         # نشون بده چقدر از full model پایین‌تره
#         diff = f"({(ndcg10 - full_ndcg)*100:+.1f}%)" if full_ndcg and r['variant'] != 'Full Model' else ""
#
#         print(f"  {r['variant']:<23} {ndcg10:>10.4f} {ndcg20:>10.4f} "
#               f"{rec10:>10.4f} {rec20:>10.4f} {epoch:>6}  {diff}")
#
#     print("=" * 80)
#
#     # Save results
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_path = os.path.join(RESULTS_DIR, f"ablation_results_{ts}.json")
#     with open(out_path, 'w') as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\nResults saved to: {out_path}")
#
#
# if __name__ == "__main__":
#     main()
"""
Ablation Study - ProtoNet Clean Project
=============================================================
4 variant رو اجرا می‌کنه و نتایج رو مقایسه می‌کنه:

1. Full Model         (social + POI features + hard negatives)
2. No Social          (بدون social neighbors)
3. No POI Features    (بدون lat/lon/category)
4. No Hard Negatives  (random negatives به جای hard)

هر variant یه cache جدا داره تا با هم تداخل نداشته باشن
=============================================================
"""

import os
import torch
import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict

# ===================== PATHS =====================
import pathlib
# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Volumes/Extreme SSD/Model/Tourism_ProtoNet")

# PROCESSED_DIR = str(_BASE / "processed")
# PROCESSED_DIR = str(_BASE / "processed_japan_v2")
PROCESSED_DIR  = str(_BASE / "processed_asia_v1")

RESULTS_DIR   = str(_BASE / "ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.append(str(_BASE))

from src.step2_dataset import (
    TourismMetaDataset, collate_meta_tasks,
    load_processed_data, build_poi_feature_matrix
)
from src.step3_model import ProtoNetGNN
from src.step4_train import train_one_epoch, EarlyStopping
from src.utils import compute_metrics

# ===================== BASE CONFIG =====================
BASE_CONFIG = {
    'embed_dim': 64,
    'dropout': 0.5,
    'max_vocab_size': 100_000,
    'temperature': 0.05,
    'support_k': 15,
    'query_size': 15,
    'lr': 0.001,
    'weight_decay': 2e-4,
    'clip_norm': 1.0,
    'batch_size': 4,
    'n_epochs': 15,
    'patience': 5,
    'eval_every': 1,
    'top_k_list': [10, 20],
    'seed': 42,
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'

}

# ===================== 4 VARIANTS =====================
VARIANTS = [
    {
        'name': 'Full Model',
        'tag': 'full',
        'use_social': True,
        'use_poi_features': True,
        'use_hard_negatives': True,
    },
    {
        'name': 'No Social',
        'tag': 'no_social',
        'use_social': False,
        'use_poi_features': True,
        'use_hard_negatives': True,
    },
    {
        'name': 'No POI Features',
        'tag': 'no_poi',
        'use_social': True,
        'use_poi_features': False,
        'use_hard_negatives': True,
    },
    {
        'name': 'No Hard Negatives',
        'tag': 'no_hard_neg',
        'use_social': True,
        'use_poi_features': True,
        'use_hard_negatives': False,
    },
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_full(model, val_loader, device, config, max_batches=40):
    model.eval()
    all_metrics = defaultdict(list)
    with torch.no_grad():
        for idx, tasks_batch in enumerate(val_loader):
            if idx >= max_batches:
                break
            metrics = compute_metrics(model, tasks_batch, device, config['top_k_list'])
            for k, v in metrics.items():
                all_metrics[k].append(v)
    return {k: np.mean(vs) for k, vs in all_metrics.items()}


def run_variant(variant, checkins, pois, mappings, splits, social_adj,
                poi_features_tensor, n_venues, poi_feature_dim, device):
    """یک variant رو train و eval می‌کنه"""

    print(f"\n{'='*60}")
    print(f"Running: {variant['name']}")
    print(f"{'='*60}")

    set_seed(BASE_CONFIG['seed'])

    cache_train = os.path.join(PROCESSED_DIR, f"cache_ablation_{variant['tag']}_train.pkl")
    cache_val   = os.path.join(PROCESSED_DIR, f"cache_ablation_{variant['tag']}_val.pkl")

    # Delete old cache to ensure correct variant is built
    import pathlib
    for cp in [cache_train, cache_val]:
        if pathlib.Path(cp).exists():
            pathlib.Path(cp).unlink()
            print(f"  Deleted old cache: {pathlib.Path(cp).name}")

    # اگه no_social باشه، social_adj خالی پاس می‌دیم
    social_adj_use = social_adj if variant['use_social'] else {}

    # اگه no_hard_neg باشه، pois_df پاس نمی‌دیم
    pois_df_use = pois if variant['use_hard_negatives'] else None

    train_ds = TourismMetaDataset(
        splits['train_users'], checkins, social_adj_use,
        BASE_CONFIG['support_k'], BASE_CONFIG['query_size'], n_venues,
        cache_path=cache_train, shuffle_support=True,
        pois_df=pois_df_use
    )

    val_ds = TourismMetaDataset(
        splits['val_users'], checkins, social_adj_use,
        BASE_CONFIG['support_k'], BASE_CONFIG['query_size'], n_venues,
        cache_path=cache_val, shuffle_support=False,
        pois_df=pois_df_use
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BASE_CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_meta_tasks, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BASE_CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_meta_tasks, num_workers=0
    )

    # اگه no_poi باشه، poi_features_tensor پاس نمی‌دیم
    poi_feat_use = poi_features_tensor if variant['use_poi_features'] else None
    # poi_feature_dim برای no_poi باید کوچیک‌تر باشه (فقط ID embedding)
    # ولی چون مدل از combiner استفاده می‌کنه، None پاس می‌دیم
    feat_dim_use = poi_feature_dim if variant['use_poi_features'] else poi_feature_dim

    model = ProtoNetGNN(
        n_venues=n_venues,
        poi_feature_dim=feat_dim_use,
        embed_dim=BASE_CONFIG['embed_dim'],
        dropout=BASE_CONFIG['dropout'],
        max_vocab_size=BASE_CONFIG['max_vocab_size'],
        temperature=BASE_CONFIG['temperature'],
        poi_features_tensor=poi_feat_use
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=BASE_CONFIG['lr'],
        weight_decay=BASE_CONFIG['weight_decay']
    )

    early_stop = EarlyStopping(patience=BASE_CONFIG['patience'])
    best_ndcg = -np.inf
    best_metrics = {}
    best_epoch = 0

    history = []

    for epoch in range(1, BASE_CONFIG['n_epochs'] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, BASE_CONFIG)
        val_metrics = evaluate_full(model, val_loader, device, BASE_CONFIG)
        elapsed = time.time() - t0

        cur_ndcg = val_metrics.get('NDCG@10', 0)
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})

        print(f"  Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"NDCG@10: {cur_ndcg:.4f} | NDCG@20: {val_metrics.get('NDCG@20', 0):.4f} | "
              f"Time: {elapsed:.1f}s")

        if cur_ndcg > best_ndcg:
            best_ndcg = cur_ndcg
            best_metrics = val_metrics.copy()
            best_epoch = epoch

        if early_stop.step(cur_ndcg, epoch):
            break

    print(f"\n  ✓ Best: epoch={best_epoch}, NDCG@10={best_ndcg:.4f}")

    return {
        'variant': variant['name'],
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'history': history,
    }


def main():
    print("=" * 60)
    print("ABLATION STUDY - ProtoNet Tourism Cold-Start")
    print("=" * 60)
    print(f"Variants: {[v['name'] for v in VARIANTS]}")

    device = torch.device(BASE_CONFIG['device'])
    print(f"Device: {device}")

    # Load data once
    print("\nLoading Japan v2 data...")
    checkins, pois, mappings, splits, social_adj = load_processed_data(PROCESSED_DIR)

    n_venues = len(mappings['venue2id'])
    n_cats = len(mappings.get('cat2id', {}))
    poi_feature_dim = 2 + n_cats + 1

    poi_features_np = build_poi_feature_matrix(pois, n_venues, mappings)
    poi_features_tensor = torch.tensor(poi_features_np, dtype=torch.float32)

    print(f"  n_venues={n_venues:,} | n_cats={n_cats} | poi_dim={poi_feature_dim}")

    # Run all variants
    all_results = []
    for variant in VARIANTS:
        result = run_variant(
            variant, checkins, pois, mappings, splits, social_adj,
            poi_features_tensor, n_venues, poi_feature_dim, device
        )
        all_results.append(result)

    # ===================== PRINT SUMMARY TABLE =====================
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Variant':<25} {'NDCG@10':>10} {'NDCG@20':>10} {'Recall@10':>10} {'Recall@20':>10} {'Epoch':>6}")
    print("-" * 80)

    full_ndcg = None
    for r in all_results:
        m = r['best_metrics']
        ndcg10 = m.get('NDCG@10', 0)
        ndcg20 = m.get('NDCG@20', 0)
        rec10  = m.get('Recall@10', 0)
        rec20  = m.get('Recall@20', 0)
        epoch  = r['best_epoch']

        if r['variant'] == 'Full Model':
            full_ndcg = ndcg10

        # نشون بده چقدر از full model پایین‌تره
        diff = f"({(ndcg10 - full_ndcg)*100:+.1f}%)" if full_ndcg and r['variant'] != 'Full Model' else ""

        print(f"  {r['variant']:<23} {ndcg10:>10.4f} {ndcg20:>10.4f} "
              f"{rec10:>10.4f} {rec20:>10.4f} {epoch:>6}  {diff}")

    print("=" * 80)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ablation_results_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
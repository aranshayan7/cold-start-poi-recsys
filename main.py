# """
# main.py - Prototypical Networks Clean Project
# """
#
# import os
# import torch
# import numpy as np
# import time
# from datetime import datetime
#
# from config import CONFIG
#
# # Correct imports
# from src.step2_dataset import (
#     TourismMetaDataset,
#     collate_meta_tasks,
#     load_processed_data,
#     build_poi_feature_matrix
# )
# from src.step3_model import ProtoNetGNN
# from src.step4_train import train_one_epoch, EarlyStopping   # ← EarlyStopping اضافه شد
# from src.utils import compute_metrics
#
#
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#
#
# def main():
#     print("=" * 80)
#     print("Prototypical Networks - Clean Project")
#     print("=" * 80)
#
#     set_seed(CONFIG['seed'])
#     os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
#     os.makedirs(CONFIG['log_dir'], exist_ok=True)
#
#     device = torch.device(CONFIG['device'])
#     print(f"Device: {device}")
#
#     # Load Data
#     print("\nLoading Japan data...")
#     checkins, pois, mappings, splits, social_adj = load_processed_data(CONFIG['processed_dir'])
#
#     n_venues = len(mappings['venue2id'])
#     n_cats = len(mappings.get('cat2id', {}))
#     poi_feature_dim = 2 + n_cats + 1
#
#     poi_features_np = build_poi_feature_matrix(pois, n_venues, mappings)
#     poi_features_tensor = torch.tensor(poi_features_np, dtype=torch.float32)
#
#     # Datasets
#     train_ds = TourismMetaDataset(
#         splits['train_users'], checkins, social_adj,
#         CONFIG['support_k'], CONFIG['query_size'], n_venues,
#         cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_train.pkl'),
#         pois_df=pois
#     )
#
#     val_ds = TourismMetaDataset(
#         splits['val_users'], checkins, social_adj,
#         CONFIG['support_k'], CONFIG['query_size'], n_venues,
#         cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_val.pkl'),
#         pois_df=pois, shuffle_support=False
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
#         collate_fn=collate_meta_tasks, num_workers=0, drop_last=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
#         collate_fn=collate_meta_tasks, num_workers=0
#     )
#
#     # Model
#     model = ProtoNetGNN(
#         n_venues=n_venues,
#         poi_feature_dim=poi_feature_dim,
#         embed_dim=CONFIG['embed_dim'],
#         dropout=CONFIG['dropout'],
#         max_vocab_size=CONFIG['max_vocab_size'],
#         temperature=CONFIG['temperature'],
#         poi_features_tensor=poi_features_tensor
#     ).to(device)
#
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
#
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=CONFIG['lr'],
#         weight_decay=CONFIG['weight_decay']
#     )
#
#     early_stop = EarlyStopping(patience=CONFIG['patience'])
#     best_ndcg = -np.inf
#
#     print("\nStarting Prototypical Networks Training...\n")
#
#     for epoch in range(1, CONFIG['n_epochs'] + 1):
#         t0 = time.time()
#         train_loss = train_one_epoch(model, optimizer, train_loader, device, CONFIG)
#
#         val_metrics = {}
#         if epoch % CONFIG['eval_every'] == 0:
#             val_batch = next(iter(val_loader))
#             val_metrics = compute_metrics(model, val_batch, device, CONFIG['top_k_list'])
#
#         elapsed = time.time() - t0
#
#         print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}/{CONFIG['n_epochs']}] | "
#               f"Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f}")
#
#         for k, v in val_metrics.items():
#             print(f"  Val {k}: {v:.4f}")
#
#         cur_ndcg = val_metrics.get('NDCG@10', 0)
#         if cur_ndcg > best_ndcg:
#             best_ndcg = cur_ndcg
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'ndcg': cur_ndcg,
#             }, os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt'))
#             print(f"  >>> Best model saved! NDCG@10: {cur_ndcg:.4f}")
#
#         if early_stop.step(cur_ndcg, epoch):
#             break
#
#     print(f"\nTraining Finished! Best NDCG@10 = {best_ndcg:.4f}")
#
#
# if __name__ == "__main__":
#     main()

"""
main.py - Prototypical Networks - Fixed Eval Protocol
eval: 100 negatives per positive, همه val batches (نه یک batch)
"""

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import json
import numpy as np_json   # برای custom encoder

class _NumpyEncoder(json.JSONEncoder):
    """تبدیل numpy types به Python types برای JSON."""
    def default(self, obj):
        if isinstance(obj, (np_json.integer,)):   return int(obj)
        if isinstance(obj, (np_json.floating,)):  return float(obj)
        if isinstance(obj, (np_json.bool_,)):     return bool(obj)
        if isinstance(obj, np_json.ndarray):      return obj.tolist()
        return super().default(obj)

from config import CONFIG

from src.step2_dataset import (
    TourismMetaDataset,
    collate_meta_tasks,
    load_processed_data,
    build_poi_feature_matrix
)
from src.step3_model import ProtoNetGNN
from src.step4_train import train_one_epoch, EarlyStopping
from src.utils import compute_metrics  # ← utils fixed شده


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, val_loader, device, config, max_batches=40):
    """Eval روی همه val batches - نه یک batch"""
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


def main():
    test_only = '--test-only' in sys.argv

    print("=" * 80)
    print("Prototypical Networks - Fixed Eval Protocol")
    if test_only:
        print("MODE: TEST ONLY (loading best checkpoint)")
    print("=" * 80)

    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    # ── JSON log init ──────────────────────────────────────────────
    _run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    _ds_tag   = CONFIG['processed_dir'].rstrip('/').split('/')[-1]
    _log_path = os.path.join(CONFIG['log_dir'], f"protonet_{_ds_tag}_{_run_ts}.json")
    _log_data = {
        "model": "ProtoNet",
        "dataset": CONFIG['processed_dir'].rstrip('/').split('/')[-1],
        "run_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: (str(v) if isinstance(v, type(torch.device('cpu'))) else v)
                   for k, v in CONFIG.items()},
        "epoch_history": [],
        "summary": {}
    }
    # ──────────────────────────────────────────────────────────────

    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")

    # print("\nLoading Japan data...")
    # print("\nLoading Asia data...")
    print("\nLoading Full Foursquare data...")
    checkins, pois, mappings, splits, social_adj = load_processed_data(CONFIG['processed_dir'])

    n_venues = len(mappings['venue2id'])
    n_cats = len(mappings.get('cat2id', {}))
    poi_feature_dim = 2 + n_cats + 1

    poi_features_np = build_poi_feature_matrix(pois, n_venues, mappings)
    poi_features_tensor = torch.tensor(poi_features_np, dtype=torch.float32)

    train_ds = TourismMetaDataset(
        splits['train_users'], checkins, social_adj,
        CONFIG['support_k'], CONFIG['query_size'], n_venues,
        # Japan
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_train.pkl'),
        # Asia
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_asia_train.pkl'),
        cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_full_train.pkl'),
        pois_df=pois
    )

    val_ds = TourismMetaDataset(
        splits['val_users'], checkins, social_adj,
        CONFIG['support_k'], CONFIG['query_size'], n_venues,
        # Japan
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_val.pkl'),
        # Asia
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_asia_val.pkl'),
        cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_full_val.pkl'),
        pois_df=pois, shuffle_support=False
    )

    test_ds = TourismMetaDataset(
        splits['test_users'], checkins, social_adj,
        CONFIG['support_k'], CONFIG['query_size'], n_venues,
        # Japan
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_test.pkl'),
        # Asia
        # cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_asia_test.pkl'),
        cache_path=os.path.join(CONFIG['processed_dir'], 'cache_protonet_full_test.pkl'),
        pois_df=pois, shuffle_support=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_meta_tasks, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_meta_tasks, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_meta_tasks, num_workers=0
    )

    model = ProtoNetGNN(
        n_venues=n_venues,
        poi_feature_dim=poi_feature_dim,
        embed_dim=CONFIG['embed_dim'],
        dropout=CONFIG['dropout'],
        max_vocab_size=CONFIG['max_vocab_size'],
        temperature=CONFIG['temperature'],
        poi_features_tensor=poi_features_tensor
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    early_stop = EarlyStopping(patience=CONFIG['patience'])
    best_ndcg = -np.inf

    if not test_only:
        print("\nStarting Training (Fixed Eval: 100 neg/pos, 40 val batches)...\n")

        for epoch in range(1, CONFIG['n_epochs'] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, optimizer, train_loader, device, CONFIG)

            val_metrics = {}
            if epoch % CONFIG['eval_every'] == 0:
                val_metrics = evaluate(model, val_loader, device, CONFIG)

            elapsed = time.time() - t0

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts}] Epoch [{epoch}/{CONFIG['n_epochs']}] | Time: {elapsed:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  Val {k}: {v:.4f}")

            cur_ndcg = val_metrics.get('NDCG@10', 0)
            _is_best  = bool(val_metrics) and (cur_ndcg > best_ndcg)
            if cur_ndcg > best_ndcg:
                best_ndcg = cur_ndcg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ndcg': cur_ndcg,
                    'metrics': val_metrics,
                }, os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt'))
                print(f"  >>> Best model saved! NDCG@10: {cur_ndcg:.4f}")

            # ── per-epoch JSON write ───────────────────────────────────
            _epoch_log = {
                "epoch": epoch,
                "timestamp": ts,
                "elapsed_seconds": round(elapsed, 2),
                "train_loss": round(train_loss, 6),
                "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
                "is_best": _is_best,
            }
            _log_data["epoch_history"].append(_epoch_log)
            with open(_log_path, 'w') as _f:
                json.dump(_log_data, _f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
            # ──────────────────────────────────────────────────────────

            if val_metrics and early_stop.step(cur_ndcg, epoch):
                break

        print(f"\nTraining Finished! Best NDCG@10 = {best_ndcg:.4f}")

    # ── test evaluation با بهترین مدل ─────────────────────────────
    ckpt_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt')
    print(f"\nLoading best model for test evaluation...\n  → {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if test_only:
        best_ndcg = float(ckpt.get('ndcg', 0))
        print(f"  Best val NDCG@10 (from checkpoint): {best_ndcg:.4f}")
    print("Running test evaluation (all test batches)...")
    test_metrics = evaluate(model, test_loader, device, CONFIG, max_batches=9999)

    print("\n" + "=" * 80)
    print("ProtoNet FINAL TEST RESULTS")
    print("=" * 80)
    for k, v in test_metrics.items():
        print(f"  Test {k}: {v:.4f}")
    print("=" * 80)
    # ──────────────────────────────────────────────────────────────

    # ── final summary JSON ─────────────────────────────────────────
    _best_ep_log = next(
        (e for e in reversed(_log_data["epoch_history"]) if e.get("is_best")), {}
    )
    _log_data["summary"] = {
        "total_epochs_run": int(len(_log_data["epoch_history"])),
        "best_epoch": int(early_stop.best_epoch),
        "early_stopped": bool(early_stop.counter >= CONFIG['patience']),
        "best_val_ndcg10": float(round(float(best_ndcg), 6)),
        "best_val_metrics": _best_ep_log.get("val_metrics", {}),
    }
    _log_data["test_metrics"] = {k: round(float(v), 6) for k, v in test_metrics.items()}
    _log_data["run_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_log_path, 'w') as _f:
        json.dump(_log_data, _f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"JSON log saved → {_log_path}")
    # ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
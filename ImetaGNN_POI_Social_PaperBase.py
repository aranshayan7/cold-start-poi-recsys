"""
IMETA-GNN + POI Features + Social Context - Closer to Paper
"""

import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np, json, time, pickle

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
from datetime import datetime
from collections import defaultdict
import pathlib

# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Volumes/Extreme SSD/Model/Tourism_ProtoNet")

PROCESSED_DIR  = str(_BASE / "processed_japan_v2")
CHECKPOINT_DIR = str(_BASE / "checkpoints_imetagnn_poi_social_japan")
# PROCESSED_DIR  = str(_BASE / "processed_asia_v1")
# CHECKPOINT_DIR = str(_BASE / "checkpoints_imetagnn_poi_social_asia")

RESULTS_DIR = str(_BASE / "IMETAGNN_SOCIAL_POI_JAPAN")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    'alpha':        0.01,      # دقیقاً طبق مقاله
    'beta':         0.001,     # دقیقاً طبق مقاله
    'T':            4,
    'support_k':    10,
    'query_size':   25,
    'embed_dim':    64,
    'dropout':      0.3,
    'max_vocab':    100_000,
    'temperature':  0.05,
    'n_categories': 64,
    'social_weight': 0.25,
    'n_epochs':     20,
    'patience':     6,
    'top_k_list':   [10, 20],
    'seed':         42,
    # 'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'

}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class IMETAGNNPOISocial(nn.Module):
    def __init__(self, n_items, poi_features, embed_dim=64,
                 dropout=0.3, max_vocab=100_000, temperature=0.05,
                 n_categories=64, social_weight=0.25):
        super().__init__()
        self.n_items = n_items
        self.max_vocab = max_vocab
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.social_weight = social_weight

        self.item_emb = nn.Embedding(max_vocab, embed_dim)

        poi_input_dim = 2 + n_categories
        self.poi_encoder = nn.Sequential(
            nn.Linear(poi_input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.combiner = nn.Linear(embed_dim * 2, embed_dim)
        self.user_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.register_buffer('poi_features', poi_features)

        # Initialization
        nn.init.xavier_uniform_(self.item_emb.weight, gain=0.01)
        for m in [self.poi_encoder, self.user_encoder]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.combiner.weight, gain=0.1)
        nn.init.zeros_(self.combiner.bias)

    def get_item_emb(self, item_ids):
        hashed = item_ids % self.max_vocab
        id_emb = self.item_emb(hashed)
        valid_ids = item_ids.clamp(0, self.poi_features.shape[0] - 1)
        poi_feat = self.poi_features[valid_ids]
        poi_emb = self.poi_encoder(poi_feat)
        combined = F.relu(self.combiner(torch.cat([id_emb, poi_emb], dim=-1)))
        return F.normalize(combined, p=2, dim=-1)

    def get_user_emb(self, support_ids):
        support_embs = self.get_item_emb(support_ids)
        pooled = support_embs.mean(dim=0)
        user_emb = self.user_encoder(pooled)
        return F.normalize(user_emb, p=2, dim=-1)

    def get_user_emb_social(self, support_ids, friend_supports):
        user_emb = self.get_user_emb(support_ids)
        if friend_supports:
            friend_embs = [self.get_user_emb(fs)
                           for fs in friend_supports if len(fs) > 0]
            if friend_embs:
                friend_proto = torch.stack(friend_embs).mean(dim=0)
                user_emb = ((1 - self.social_weight) * user_emb +
                            self.social_weight * friend_proto)
                user_emb = F.normalize(user_emb, p=2, dim=-1)
        return user_emb

    def score(self, user_emb, item_ids):
        return torch.matmul(self.get_item_emb(item_ids), user_emb) / self.temperature

    # === Loss مطابق مقاله (Binary Cross Entropy) ===
    def bce_loss(self, user_emb, pos_ids, neg_ids):
        pos_s = self.score(user_emb, pos_ids)
        neg_s = self.score(user_emb, neg_ids)
        pos_loss = F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s), reduction='mean')
        neg_loss = F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s), reduction='mean')
        return pos_loss + neg_loss


# === Dataset (بدون تغییر) ===
class IMETAPOIDataset(Dataset):
    def __init__(self, user_list, checkins, social_adj, n_items,
                 support_k=10, query_size=25, cache_path=None):
        self.n_items = n_items
        self.support_k = support_k
        self.query_size = query_size
        self.social_adj = social_adj

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tasks = pickle.load(f)
        else:
            print("  Building IMETA-GNN+POI tasks...")
            self.tasks = self._build(user_list, checkins)
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.tasks, f)
        print(f"  Valid tasks: {len(self.tasks):,}")

    def _build(self, user_list, checkins):
        user_set = set(user_list)
        grouped = (checkins[checkins['user_idx'].isin(user_set)]
                   .groupby('user_idx')['venue_idx']
                   .apply(list).to_dict())
        tasks = []
        rng = np.random.RandomState(42)
        for uid, venues in grouped.items():
            if len(venues) < self.support_k + 1:
                continue
            v = venues.copy()
            rng.shuffle(v)
            support = v[:self.support_k]
            query = [x for x in v[self.support_k:self.support_k + self.query_size]
                     if x not in set(support)]
            if not query:
                continue
            friends = self.social_adj.get(uid, [])
            friend_supports = [grouped[fid][:self.support_k]
                               for fid in friends[:3]
                               if fid in grouped and len(grouped[fid]) >= 3]
            tasks.append({'uid': uid, 'support': support, 'query': query,
                          'visited': set(v), 'friend_supports': friend_supports})
        return tasks

    def __len__(self): return len(self.tasks)

    def __getitem__(self, idx):
        t = self.tasks[idx]
        support = t['support'].copy()
        np.random.shuffle(support)
        negs = []
        while len(negs) < len(t['query']):
            neg = np.random.randint(0, self.n_items)
            if neg not in t['visited']:
                negs.append(neg)
        return {
            'support':         torch.tensor(support, dtype=torch.long),
            'query':           torch.tensor(t['query'], dtype=torch.long),
            'neg':             torch.tensor(negs, dtype=torch.long),
            'friend_supports': [torch.tensor(fs, dtype=torch.long)
                                for fs in t['friend_supports']],
        }


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0]}


def maml_inner_update(model, support, neg, alpha):
    # Split support: first half as user context, second half as inner positives
    # Prevents trivial collapse (same items used for both user_emb and positives)
    n = len(support)
    mid = max(1, n // 2)
    ctx = support[:mid]           # user context
    inner_pos = support[mid:]     # inner positives (different from context)
    if len(inner_pos) == 0:
        inner_pos = support       # fallback for very short support

    user_emb = model.get_user_emb(ctx)
    loss = model.bce_loss(user_emb, inner_pos, neg)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=True, allow_unused=True)
    fw = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        if grad is not None:
            fw[name] = param - alpha * grad.clamp(-1.0, 1.0)
        else:
            fw[name] = param
    return fw


def get_item_emb_fw(item_ids, fw, model):
    hashed = item_ids % model.max_vocab
    id_emb = F.embedding(hashed, fw['item_emb.weight'])
    valid_ids = item_ids.clamp(0, model.poi_features.shape[0] - 1)
    poi_feat = model.poi_features[valid_ids]
    h = F.relu(F.linear(poi_feat, fw['poi_encoder.0.weight'], fw['poi_encoder.0.bias']))
    poi_emb = F.linear(h, fw['poi_encoder.2.weight'], fw['poi_encoder.2.bias'])
    cat = torch.cat([id_emb, poi_emb], dim=-1)
    combined = F.relu(F.linear(cat, fw['combiner.weight'], fw['combiner.bias']))
    return F.normalize(combined, p=2, dim=-1)


def query_loss_fw(model, fw, support, query, neg):
    sup_embs = get_item_emb_fw(support, fw, model)
    pooled = sup_embs.mean(dim=0)
    h = F.relu(F.linear(pooled, fw['user_encoder.0.weight'], fw['user_encoder.0.bias']))
    user_emb = F.normalize(
        F.linear(h, fw['user_encoder.2.weight'], fw['user_encoder.2.bias']),
        p=2, dim=-1)
    pos_s = (torch.matmul(get_item_emb_fw(query, fw, model), user_emb)
             / model.temperature).clamp(-10, 10)
    neg_s = (torch.matmul(get_item_emb_fw(neg, fw, model), user_emb)
             / model.temperature).clamp(-10, 10)
    return F.softplus(-pos_s).mean() + F.softplus(neg_s).mean()

def compute_metrics(model, loader, device, top_k_list, max_batches=40):
    model.eval()
    metrics = defaultdict(list)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            for i in range(len(batch['support'])):
                support = batch['support'][i].to(device)
                query = batch['query'][i].to(device)
                friend_supports = [fs.to(device) for fs in batch['friend_supports'][i]]

                if len(support) < 3 or len(query) == 0:
                    continue

                user_emb = model.get_user_emb_social(support, friend_supports)
                true_pos = set(query.cpu().numpy().tolist())
                excluded = true_pos | set(support.cpu().numpy().tolist())

                # n_neg = 400
                n_neg = min(len(true_pos) * 100, 2000)  # standard: 100 per positive
                negs = []
                visited = excluded.copy()
                attempts = 0
                while len(negs) < n_neg and attempts < 3000:
                    neg = np.random.randint(0, model.n_items)
                    if neg not in visited:
                        negs.append(neg)
                        visited.add(neg)
                    attempts += 1

                candidates = list(true_pos) + negs
                np.random.shuffle(candidates)   # prevent ordering bias → fake NDCG=1.0
                cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
                scores = model.score(user_emb, cand_t)

                # Debug
                if idx == 0 and i == 0:
                    pos_mean = scores[:len(true_pos)].mean().item()
                    neg_mean = scores[len(true_pos):].mean().item()
                    print(f"Debug - Pos mean: {pos_mean:.4f} | Neg mean: {neg_mean:.4f}")

                for k in top_k_list:
                    k_actual = min(k, len(candidates))
                    topk_idx = torch.topk(scores, k=k_actual).indices.cpu().numpy()
                    top_items = [candidates[j] for j in topk_idx]

                    hits = len(set(top_items) & true_pos)

                    metrics[f'Precision@{k}'].append(hits / k_actual)
                    metrics[f'Recall@{k}'].append(hits / len(true_pos))

                    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_items) if item in true_pos)
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_pos), k_actual)))
                    metrics[f'NDCG@{k}'].append(dcg / idcg if idcg > 0 else 0.0)

    return {k: np.mean(v) for k, v in metrics.items() if v}

def train_epoch(model, optimizer, loader, device, config):
    model.train()
    total_loss, n_batches = 0.0, 0
    alpha = config['alpha']
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, device=device)
        valid = 0
        for i in range(len(batch['support'])):
            sv = batch['support'][i].to(device)
            qv = batch['query'][i].to(device)
            nv = batch['neg'][i].to(device)
            if len(sv) < 3 or len(qv) == 0:
                continue
            try:
                fw = maml_inner_update(model, sv, nv, alpha)
                q_loss = query_loss_fw(model, fw, sv, qv, nv)
                if not (torch.isnan(q_loss) or torch.isinf(q_loss)):
                    meta_loss = meta_loss + q_loss
                    valid += 1
            except Exception:
                continue
        if valid == 0:
            continue
        avg_loss = meta_loss / valid
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            optimizer.zero_grad()
            continue
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += avg_loss.item()
        n_batches += 1
        if (batch_idx + 1) % 100 == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"    [{ts}] Batch [{batch_idx+1}/{len(loader)}] | "
                  f"Loss: {total_loss/n_batches:.4f}")
    return total_loss / max(n_batches, 1)

def build_poi_features(pois_df, mappings, n_categories=64):
    n_venues = len(mappings['venue2id'])
    features = np.zeros((n_venues, 2 + n_categories), dtype=np.float32)
    lats, lons = pois_df['latitude'].values, pois_df['longitude'].values
    lat_mean, lat_std = lats.mean(), lats.std() + 1e-8
    lon_mean, lon_std = lons.mean(), lons.std() + 1e-8
    for _, row in pois_df.iterrows():
        vid = int(row['venue_idx'])  # USE venue_idx DIRECTLY
        if vid >= n_venues:
            continue
        features[vid, 0] = (row['latitude'] - lat_mean) / lat_std
        features[vid, 1] = (row['longitude'] - lon_mean) / lon_std
        cat_id = int(row.get('category_idx', 0)) % n_categories
        features[vid, 2 + cat_id] = 1.0
    return torch.tensor(features, dtype=torch.float32)


def main():
    print("=" * 65)
    print("IMETA-GNN + POI Features + Social Context")
    print(f"alpha={CONFIG['alpha']} | beta={CONFIG['beta']} | T={CONFIG['T']}")
    print(f"social_weight={CONFIG['social_weight']} | temp={CONFIG['temperature']}")
    print("=" * 65)

    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")

    import pandas as pd
    checkins = pd.read_parquet(os.path.join(PROCESSED_DIR, "checkins.parquet"))
    pois_df  = pd.read_parquet(os.path.join(PROCESSED_DIR, "pois.parquet"))
    with open(os.path.join(PROCESSED_DIR, "mappings.pkl"), 'rb') as f:
        mappings = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "social_adj.pkl"), 'rb') as f:
        social_adj = pickle.load(f)

    n_items = len(mappings['venue2id'])
    print(f"n_items={n_items:,} | train={len(splits['train_users']):,} | "
          f"val={len(splits['val_users']):,} | test={len(splits['test_users']):,}")

    print("Building POI features...")
    poi_features = build_poi_features(pois_df, mappings, CONFIG['n_categories']).to(device)
    print(f"POI features shape: {poi_features.shape}")

    # DEBUG
    print("venue2id sample:", list(mappings['venue2id'].items())[:3])
    print("pois venue_id sample:", pois_df['venue_id'].head(3).tolist())
    print("pois venue_idx sample:", pois_df['venue_idx'].head(3).tolist())
    non_zero = (poi_features.cpu().abs().sum(dim=1) > 0).sum().item()
    print(f"Venues with non-zero features: {non_zero} / {poi_features.shape[0]}")
    cp = "cache_imetagnn_poi_social"
    train_ds = IMETAPOIDataset(splits['train_users'], checkins, social_adj, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, f'{cp}_train.pkl'))
    val_ds   = IMETAPOIDataset(splits['val_users'],   checkins, social_adj, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, f'{cp}_val.pkl'))
    test_ds  = IMETAPOIDataset(splits['test_users'],  checkins, social_adj, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, f'{cp}_test.pkl'))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['T'], shuffle=True,
                              collate_fn=collate, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['T'], shuffle=False,
                              collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['T'], shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = IMETAGNNPOISocial(
        n_items=n_items, poi_features=poi_features,
        embed_dim=CONFIG['embed_dim'], dropout=CONFIG['dropout'],
        max_vocab=CONFIG['max_vocab'], temperature=CONFIG['temperature'],
        n_categories=CONFIG['n_categories'], social_weight=CONFIG['social_weight'],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG['beta'], weight_decay=1e-5)
    best_ndcg, best_metrics, patience_counter = -np.inf, {}, 0
    epoch_history = []          # ← epoch-by-epoch log

    print("\nTraining IMETA-GNN+POI+Social...\n")
    for epoch in range(1, CONFIG['n_epochs'] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, device, CONFIG)
        val_metrics = compute_metrics(model, val_loader, device, CONFIG['top_k_list'])
        elapsed = time.time() - t0
        ndcg = val_metrics['NDCG@10']
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Epoch [{epoch}/{CONFIG['n_epochs']}] | Time: {elapsed:.1f}s")
        print(f"  Loss: {train_loss:.4f} | NDCG@10: {ndcg:.4f} | "
              f"NDCG@20: {val_metrics['NDCG@20']:.4f}")
        _is_best = ndcg > best_ndcg + 1e-4
        if _is_best:
            best_ndcg = ndcg
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'ndcg': ndcg},
                       os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  >>> Best model saved! NDCG@10={ndcg:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping! Best NDCG@10={best_ndcg:.4f}")
                epoch_history.append({
                    "epoch": epoch, "timestamp": ts,
                    "elapsed_seconds": round(elapsed, 2),
                    "train_loss": round(train_loss, 6),
                    "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
                    "is_best": False,
                })
                break

        epoch_history.append({
            "epoch": epoch, "timestamp": ts,
            "elapsed_seconds": round(elapsed, 2),
            "train_loss": round(train_loss, 6),
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "is_best": _is_best,
        })

    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_metrics = compute_metrics(model, test_loader, device,
                                   CONFIG['top_k_list'], max_batches=9999)

    print("\n" + "=" * 65)
    print("IMETA-GNN+POI+Social FINAL RESULTS")
    print("=" * 65)
    print(f"  Val  NDCG@10: {best_ndcg:.4f}")
    print(f"  Test NDCG@10: {test_metrics['NDCG@10']:.4f}")
    print(f"  Test NDCG@20: {test_metrics['NDCG@20']:.4f}")
    print(f"  Test Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"  Test Recall@20: {test_metrics['Recall@20']:.4f}")
    print("=" * 65)

    results = {
        'model': 'IMETA-GNN+POI+Social',
        'dataset': PROCESSED_DIR.rstrip('/').split('/')[-1],
        'config': CONFIG,
        'epoch_history': epoch_history,
        'summary': {
            'total_epochs_run': len(epoch_history),
            'best_val_ndcg10': round(best_ndcg, 6),
            'best_val_metrics': best_metrics,
        },
        'val_metrics': best_metrics,
        'test_metrics': test_metrics,
    }
    _ds_tag = PROCESSED_DIR.rstrip('/').split('/')[-1]   # e.g. processed_asia_v1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"imetagnn_poi_social_{_ds_tag}_{ts}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
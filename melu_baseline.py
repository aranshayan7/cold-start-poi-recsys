"""
MeLU: Meta-Learned User Preference Estimator
=============================================================
Based on: "MeLU: Meta-Learned User Preference Estimator 
           for Cold-Start Recommendation" (KDD 2019)

Simple MAML-based recommendation:
- No GNN, no social network, no POI features
- Just user/item embeddings + MAML adaptation
- This is the standard baseline for cold-start papers

We implement it on our Japan v2 dataset for fair comparison
=============================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
import time
import pickle
from datetime import datetime
from collections import defaultdict

import pathlib
# _BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
_BASE = pathlib.Path(r"/Volumes/Extreme SSD/Model/Tourism_ProtoNet")

# PROCESSED_DIR  = str(_BASE / "processed_japan_v2")
# CHECKPOINT_DIR = str(_BASE / "checkpoints_melu")
# RESULTS_DIR    = str(_BASE / "MELU_JAPAN")
#
PROCESSED_DIR  = str(_BASE / "processed_asia_v1")
CHECKPOINT_DIR = str(_BASE / "checkpoints_melu_asia")
RESULTS_DIR    = str(_BASE / "MELU_ASIA")

# PROCESSED_DIR  = str(_BASE / "processed")
# CHECKPOINT_DIR = str(_BASE / "checkpoints_melu_full")
# RESULTS_DIR    = str(_BASE / "MELU_FULL")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== CONFIG =====================
CONFIG = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'alpha': 0.01,       # inner LR
    'beta': 0.001,       # meta LR
    'inner_steps': 1,
    'support_k': 15,
    'query_size': 15,
    'batch_size': 4,     # n tasks per meta-update
    'n_epochs': 15,
    'patience': 5,
    'top_k_list': [10, 20],
    'seed': 42,
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',

    'max_vocab': 100_000,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ===================== MODEL =====================
class MeLUModel(nn.Module):
    """
    MeLU: Simple embedding-based model for cold-start
    
    Architecture (as in paper):
    - User preference: mean of support item embeddings
    - Score: dot product of user preference and item embedding
    - No social network, no geographic features
    """

    def __init__(self, n_items, embed_dim=64, hidden_dim=64,
                 max_vocab=100_000, temperature=0.05):
        super().__init__()
        self.n_items = n_items
        self.max_vocab = max_vocab
        self.temperature = temperature

        # Item embeddings
        self.item_embedding = nn.Embedding(max_vocab, embed_dim)

        # User preference network
        self.user_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        nn.init.xavier_uniform_(self.item_embedding.weight, gain=0.1)

    def get_item_emb(self, item_ids):
        hashed = item_ids % self.max_vocab
        emb = self.item_embedding(hashed)
        return F.normalize(emb, p=2, dim=-1)

    def get_user_emb(self, support_ids):
        """User preference = mean of support item embeddings"""
        support_embs = self.get_item_emb(support_ids)
        user_emb = support_embs.mean(dim=0)
        user_emb = self.user_net(user_emb)
        return F.normalize(user_emb, p=2, dim=-1)

    def score(self, user_emb, item_ids):
        item_embs = self.get_item_emb(item_ids)
        return torch.matmul(item_embs, user_emb) / self.temperature

    def loss(self, user_emb, pos_ids, neg_ids):
        pos_scores = self.score(user_emb, pos_ids)
        neg_scores = self.score(user_emb, neg_ids)
        neg_exp = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)
        logits = torch.cat([pos_scores.unsqueeze(1), neg_exp], dim=1)
        targets = torch.zeros(len(pos_scores), dtype=torch.long,
                              device=user_emb.device)
        return F.cross_entropy(logits, targets)


# ===================== DATASET =====================
class MeLUDataset(Dataset):
    """Simple dataset for MeLU - no social, no POI features"""

    def __init__(self, user_list, checkins, n_items,
                 support_k=15, query_size=15,
                 cache_path=None):
        self.n_items = n_items
        self.support_k = support_k
        self.query_size = query_size

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tasks = pickle.load(f)
        else:
            print("  Building MeLU tasks...")
            self.tasks = self._build(user_list, checkins)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path)
                            if os.path.dirname(cache_path) else '.', exist_ok=True)
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
            tasks.append({
                'uid': uid,
                'support': support,
                'query': query,
                'visited': set(v),
            })
        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        t = self.tasks[idx]
        support = t['support'].copy()
        np.random.shuffle(support)

        visited = t['visited']
        negs = []
        while len(negs) < len(t['query']):
            neg = np.random.randint(0, self.n_items)
            if neg not in visited:
                negs.append(neg)

        return {
            'support': torch.tensor(support, dtype=torch.long),
            'query':   torch.tensor(t['query'], dtype=torch.long),
            'neg':     torch.tensor(negs, dtype=torch.long),
        }


def collate(batch):
    return {
        'support': [b['support'] for b in batch],
        'query':   [b['query']   for b in batch],
        'neg':     [b['neg']     for b in batch],
    }


# ===================== MAML INNER UPDATE =====================
def inner_update(model, support, query, neg, alpha, device):
    """Standard MAML inner loop on support set"""
    user_emb = model.get_user_emb(support)
    inner_loss = model.loss(user_emb, support, neg)

    grads = torch.autograd.grad(
        inner_loss, list(model.parameters()),
        create_graph=True, allow_unused=True
    )

    fast_weights = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        fast_weights[name] = param - alpha * grad if grad is not None else param

    return fast_weights


def query_loss_fw(model, fast_weights, support, query, neg, device):
    """Query loss with fast weights"""
    hashed = support % model.max_vocab
    id_emb = F.embedding(hashed, fast_weights['item_embedding.weight'])
    id_emb = F.normalize(id_emb, p=2, dim=-1)
    user_emb_raw = id_emb.mean(dim=0)

    # Apply user_net with fast weights
    w1 = fast_weights['user_net.0.weight']
    b1 = fast_weights['user_net.0.bias']
    w2 = fast_weights['user_net.2.weight']
    b2 = fast_weights['user_net.2.bias']

    h = F.relu(F.linear(user_emb_raw, w1, b1))
    user_emb = F.normalize(F.linear(h, w2, b2), p=2, dim=-1)

    def get_emb_fw(ids):
        hashed = ids % model.max_vocab
        emb = F.embedding(hashed, fast_weights['item_embedding.weight'])
        return F.normalize(emb, p=2, dim=-1)

    pos_scores = torch.matmul(get_emb_fw(query), user_emb) / model.temperature
    neg_scores = torch.matmul(get_emb_fw(neg), user_emb) / model.temperature
    neg_exp = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)
    logits = torch.cat([pos_scores.unsqueeze(1), neg_exp], dim=1)
    targets = torch.zeros(len(pos_scores), dtype=torch.long, device=device)
    return F.cross_entropy(logits, targets)


# ===================== EVAL =====================
def compute_metrics(model, loader, device, top_k_list, max_batches=40):
    model.eval()
    metrics = defaultdict(list)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            for i in range(len(batch['support'])):
                support = batch['support'][i].to(device)
                query   = batch['query'][i].to(device)

                if len(support) < 3 or len(query) == 0:
                    continue

                user_emb = model.get_user_emb(support)

                true_pos = set(query.cpu().numpy().tolist())
                sup_set  = set(support.cpu().numpy().tolist())
                excluded = true_pos | sup_set

                n_neg = min(len(true_pos) * 100, 2000)
                negs = []
                visited = set(excluded)
                attempts = 0
                while len(negs) < n_neg and attempts < n_neg * 5:
                    neg = np.random.randint(0, model.n_items)
                    if neg not in visited:
                        negs.append(neg)
                        visited.add(neg)
                    attempts += 1

                candidates = list(true_pos) + negs
                cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
                scores = model.score(user_emb, cand_t)

                for k in top_k_list:
                    topk = torch.topk(scores, k=min(k, len(candidates))).indices
                    recs = set(candidates[i] for i in topk.cpu().numpy())
                    hits = len(recs & true_pos)
                    metrics[f'Precision@{k}'].append(hits / k)
                    metrics[f'Recall@{k}'].append(hits / len(true_pos))

                    dcg  = sum(1.0 / np.log2(i + 2)
                               for i, c in enumerate(
                                   [candidates[j] for j in topk.cpu().numpy()])
                               if c in true_pos)
                    idcg = sum(1.0 / np.log2(i + 2)
                               for i in range(min(len(true_pos), k)))
                    metrics[f'NDCG@{k}'].append(dcg / idcg if idcg > 0 else 0.0)

    return {k: np.mean(v) for k, v in metrics.items()}


# ===================== TRAIN =====================
def train_epoch(model, optimizer, loader, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
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

            fw = inner_update(model, sv, qv, nv, alpha, device)
            q_loss = query_loss_fw(model, fw, sv, qv, nv, device)
            meta_loss += q_loss
            valid += 1

        if valid == 0:
            continue

        (meta_loss / valid).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += (meta_loss / valid).item()
        n_batches += 1

        if (batch_idx + 1) % 100 == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"    [{ts}] Batch [{batch_idx+1}/{len(loader)}] | "
                  f"Loss: {total_loss/n_batches:.4f}")

    return total_loss / max(n_batches, 1)


# ===================== MAIN =====================
def main():
    print("=" * 65)
    print("MeLU - Meta-Learned User Preference Estimator (KDD 2019)")
    print("Baseline for cold-start comparison")
    print("=" * 65)

    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")

    # Load data
    import pandas as pd
    checkins  = pd.read_parquet(os.path.join(PROCESSED_DIR, "checkins.parquet"))
    with open(os.path.join(PROCESSED_DIR, "mappings.pkl"), 'rb') as f:
        mappings = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    n_items = len(mappings['venue2id'])
    print(f"n_items={n_items:,} | train={len(splits['train_users']):,} | "
          f"val={len(splits['val_users']):,} | test={len(splits['test_users']):,}")

    # Datasets
    train_ds = MeLUDataset(
        splits['train_users'], checkins, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, 'cache_melu_train.pkl')
    )
    val_ds = MeLUDataset(
        splits['val_users'], checkins, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, 'cache_melu_val.pkl')
    )
    test_ds = MeLUDataset(
        splits['test_users'], checkins, n_items,
        CONFIG['support_k'], CONFIG['query_size'],
        cache_path=os.path.join(PROCESSED_DIR, 'cache_melu_test.pkl')
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                              shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                              shuffle=False, collate_fn=collate, num_workers=0)

    # Model
    model = MeLUModel(
        n_items=n_items,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        max_vocab=CONFIG['max_vocab'],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['beta'])

    best_ndcg = -np.inf
    best_metrics = {}
    patience_counter = 0
    epoch_history = []          # ← epoch-by-epoch log

    print("\nTraining MeLU...\n")

    for epoch in range(1, CONFIG['n_epochs'] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, device, CONFIG)
        val_metrics = compute_metrics(model, val_loader, device,
                                      CONFIG['top_k_list'])
        elapsed = time.time() - t0

        ndcg = val_metrics['NDCG@10']
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Epoch [{epoch}/{CONFIG['n_epochs']}] | "
              f"Time: {elapsed:.1f}s")
        print(f"  Loss: {train_loss:.4f} | NDCG@10: {ndcg:.4f} | "
              f"NDCG@20: {val_metrics['NDCG@20']:.4f}")

        _is_best = ndcg > best_ndcg + 1e-4
        if _is_best:
            best_ndcg = ndcg
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'ndcg': ndcg},
                       os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  Best saved! NDCG@10={ndcg:.4f}")
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

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_metrics = compute_metrics(model, test_loader, device,
                                   CONFIG['top_k_list'], max_batches=999)

    print("\n" + "=" * 65)
    print("MELU FINAL RESULTS")
    print("=" * 65)
    print(f"  Val  NDCG@10: {best_ndcg:.4f}")
    print(f"  Test NDCG@10: {test_metrics['NDCG@10']:.4f}")
    print(f"  Test NDCG@20: {test_metrics['NDCG@20']:.4f}")
    print(f"  Test Recall@10: {test_metrics['Recall@10']:.4f}")
    print("=" * 65)

    # Save results (with epoch history)
    results = {
        'model': 'MeLU',
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
    _ds_tag = PROCESSED_DIR.rstrip('/').split('/')[-1]   # e.g. processed_japan_v2
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"melu_{_ds_tag}_{ts}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()

"""
Step 2: Dataset for Prototypical Networks
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset
from collections import defaultdict


def _read_parquet_safe(path):
    """Read parquet with pyarrow; fall back to fastparquet on version mismatch."""
    try:
        return pd.read_parquet(path, engine='pyarrow')
    except Exception as e1:
        try:
            return pd.read_parquet(path, engine='fastparquet')
        except Exception as e2:
            # last resort: pyarrow with pre_buffer disabled
            try:
                import pyarrow.parquet as pq
                return pq.read_table(path, pre_buffer=False).to_pandas()
            except Exception:
                raise RuntimeError(
                    f"Cannot read {path}.\n"
                    f"  pyarrow error:     {e1}\n"
                    f"  fastparquet error: {e2}\n"
                    f"  Tip: pip install fastparquet"
                ) from e2


def load_processed_data(processed_dir):
    checkins = _read_parquet_safe(os.path.join(processed_dir, "checkins.parquet"))
    pois     = _read_parquet_safe(os.path.join(processed_dir, "pois.parquet"))

    with open(os.path.join(processed_dir, "mappings.pkl"), 'rb') as f:
        mappings = pickle.load(f)
    with open(os.path.join(processed_dir, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(processed_dir, "social_adj.pkl"), 'rb') as f:
        social_adj = pickle.load(f)

    return checkins, pois, mappings, splits, social_adj


def build_poi_feature_matrix(pois, n_venues, mappings):
    n_cats = len(mappings.get('cat2id', {}))
    feature_dim = 2 + n_cats + 1
    features = np.zeros((n_venues, feature_dim), dtype=np.float32)

    for _, row in pois.iterrows():
        idx = int(row['venue_idx'])
        if idx >= n_venues:
            continue
        features[idx, 0] = row.get('lat_norm', 0.0)
        features[idx, 1] = row.get('lon_norm', 0.0)
        cat_idx = int(row.get('category_idx', 0))
        if cat_idx < n_cats:
            features[idx, 2 + cat_idx] = 1.0
    return features


class TourismMetaDataset(Dataset):
    def __init__(self, user_list, checkins, social_adj,
                 support_k=15, query_size=15, n_venues=None,
                 cache_path=None, shuffle_support=True, pois_df=None):

        self.support_k = support_k
        self.query_size = query_size
        self.n_venues = n_venues
        self.social_adj = social_adj
        self.shuffle_support = shuffle_support

        # Category mapping for hard negatives
        self.venue_category = {}
        self.category_venues = defaultdict(list)
        if pois_df is not None:
            for _, row in pois_df.iterrows():
                v = int(row['venue_idx'])
                c = int(row.get('category_idx', 0))
                self.venue_category[v] = c
                self.category_venues[c].append(v)

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tasks = pickle.load(f)['tasks']
        else:
            print("  Building fixed tasks...")
            self.tasks = self._build_fixed_tasks(user_list, checkins)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'tasks': self.tasks}, f)

        print(f"  Valid tasks: {len(self.tasks):,}")

    def _build_fixed_tasks(self, user_list, checkins):
        user_set = set(user_list)
        filtered = checkins[checkins['user_idx'].isin(user_set)]
        grouped = filtered.groupby('user_idx')['venue_idx'].apply(list).to_dict()

        tasks = []
        rng = np.random.RandomState(42)

        for uid, venues in grouped.items():
            if len(venues) < self.support_k + self.query_size:
                continue

            venues_copy = venues.copy()
            rng.shuffle(venues_copy)

            support = venues_copy[:self.support_k]
            query = venues_copy[self.support_k: self.support_k + self.query_size]

            if len(query) == 0:
                continue

            support_set = set(support)
            query = [v for v in query if v not in support_set]
            if len(query) == 0:
                continue

            # Social
            friends = self.social_adj.get(uid, [])
            social_neighbors = friends[:15] if friends else []

            tasks.append({
                'user_idx': uid,
                'support_venues': support,
                'query_venues': query,
                'social_neighbors': social_neighbors,
            })

        return tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        support = task['support_venues'].copy()

        if self.shuffle_support:
            np.random.shuffle(support)

        # Hard negatives
        visited = set(support) | set(task['query_venues'])
        negatives = []
        while len(negatives) < len(task['query_venues']):
            neg = np.random.randint(0, self.n_venues)
            if neg not in visited:
                negatives.append(neg)

        return {
            'user_idx': task['user_idx'],
            'support_venues': torch.tensor(support, dtype=torch.long),
            'query_venues': torch.tensor(task['query_venues'], dtype=torch.long),
            'negative_venues': torch.tensor(negatives, dtype=torch.long),
            'social_neighbors': torch.tensor(task['social_neighbors'], dtype=torch.long),
        }


def collate_meta_tasks(batch):
    return {
        'user_idx': [item['user_idx'] for item in batch],
        'support_venues': [item['support_venues'] for item in batch],
        'query_venues': [item['query_venues'] for item in batch],
        'negative_venues': [item['negative_venues'] for item in batch],
        'social_neighbors': [item['social_neighbors'] for item in batch],
    }


if __name__ == "__main__":
    print("step2_dataset loaded successfully!")
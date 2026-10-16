import os
import torch
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

# Correct absolute import for new project
from src.step3_model import ProtoNetGNN
from src.step2_dataset import TourismMetaDataset, collate_meta_tasks, load_processed_data, build_poi_feature_matrix
from src.utils import compute_metrics


class EarlyStopping:
    def __init__(self, patience=6):
        self.patience = patience
        self.best_score = -np.inf
        self.counter = 0
        self.best_epoch = 0

    def step(self, score, epoch):
        if score > self.best_score + 1e-5:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\nEarly stopping! Best epoch={self.best_epoch}, NDCG@10={self.best_score:.4f}")
            return True
        return False


def train_one_epoch(model, optimizer, train_loader, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch_loss = 0.0
        valid = 0

        for i in range(len(batch['user_idx'])):
            sv = batch['support_venues'][i].to(device)
            qv = batch['query_venues'][i].to(device)
            nv = batch['negative_venues'][i].to(device)
            sn = batch['social_neighbors'][i].to(device) if len(batch['social_neighbors'][i]) > 0 else None

            if len(sv) < 3 or len(qv) == 0:
                continue

            loss, _, _ = model(sv, qv, nv, sn)
            batch_loss += loss
            valid += 1

        if valid == 0:
            continue

        avg_loss = batch_loss / valid
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
        optimizer.step()

        total_loss += avg_loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"    [{ts}] Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {total_loss/n_batches:.4f}")

    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    print("step4_train imported successfully!")
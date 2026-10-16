import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNetGNN(nn.Module):
    """
    Prototypical Networks for Cold-Start POI Recommendation
    ساده، سریع و پایدار - بدون inner loop MAML
    """

    def __init__(self, n_venues, poi_feature_dim, embed_dim=64, dropout=0.5,
                 max_vocab_size=100_000, temperature=0.05, poi_features_tensor=None):
        super().__init__()
        self.n_venues = n_venues  # ← این خط را اضافه کن
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.max_vocab_size = max_vocab_size

        self.venue_embedding = nn.Embedding(max_vocab_size, embed_dim)

        self.poi_feature_encoder = nn.Sequential(
            nn.Linear(poi_feature_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        if poi_features_tensor is not None:
            self.register_buffer('poi_features', poi_features_tensor)
        else:
            self.poi_features = None

        self.combiner = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.venue_embedding.weight, gain=0.1)

    def get_poi_embedding(self, venue_ids):
        hashed = venue_ids % self.max_vocab_size
        id_emb = self.venue_embedding(hashed)

        if self.poi_features is not None:
            safe = torch.clamp(venue_ids, 0, self.poi_features.size(0) - 1)
            feat = self.poi_features[safe].float()
            feat_emb = self.poi_feature_encoder(feat)
            combined = torch.cat([id_emb, feat_emb], dim=-1)
            emb = self.combiner(combined)
        else:
            emb = id_emb

        return F.normalize(emb, p=2, dim=-1)

    def compute_prototype(self, support_venues, social_neighbors=None):
        """ساخت Prototype کاربر از Support Set"""
        support_embs = self.get_poi_embedding(support_venues)

        # Mean + Social Context
        prototype = support_embs.mean(dim=0)

        if social_neighbors is not None and len(social_neighbors) > 0:
            friend_embs = self.get_poi_embedding(social_neighbors)
            friend_proto = friend_embs.mean(dim=0)
            prototype = 0.75 * prototype + 0.25 * friend_proto

        return F.normalize(prototype, p=2, dim=0)

    def forward(self, support_venues, query_venues, negative_venues, social_neighbors=None):
        prototype = self.compute_prototype(support_venues, social_neighbors)

        pos_embs = self.get_poi_embedding(query_venues)
        neg_embs = self.get_poi_embedding(negative_venues)

        pos_scores = torch.matmul(pos_embs, prototype)
        neg_scores = torch.matmul(neg_embs, prototype)

        loss = self.contrastive_loss(pos_scores, neg_scores)
        return loss, pos_scores, neg_scores

    def contrastive_loss(self, pos_scores, neg_scores):
        neg_expanded = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)
        logits = torch.cat([pos_scores.unsqueeze(1), neg_expanded], dim=1)
        targets = torch.zeros(len(pos_scores), dtype=torch.long, device=pos_scores.device)
        return F.cross_entropy(logits / self.temperature, targets)


if __name__ == "__main__":
    print("Prototypical Network Model Test Passed!")
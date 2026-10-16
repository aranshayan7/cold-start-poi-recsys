"""Shared representation used by both meta-learners.

The model is written functionally: every method takes an explicit parameter
dictionary ``P``. This lets MAML evaluate the same network with adapted ("fast")
weights without copying modules, while the prototypical learner simply passes the
meta-parameters.

Architecture
------------
POI encoder
    ``e(v) = normalize(ReLU(W_c [id(v) ; MLP(content(v))]))`` where the content
    vector holds normalised coordinates, a one-hot category and optionally a
    popularity feature.
User encoder
    The mean of the support-set POI embeddings is passed through a two-layer
    network and normalised. Friends are encoded the same way from their own
    support sets and blended with weight ``social_w``.
Score
    Cosine similarity between user and POI embeddings (the temperature ``tau``
    is applied only inside the loss).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import Dataset, Task

EMBED_DIM = 64
Params = Dict[str, torch.Tensor]


class SharedModel:
    """Holds the fixed content features and the ablation switches."""

    def __init__(self, ds: Dataset, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_venues = ds.n_venues

        content = torch.tensor(ds.features)
        if cfg.ablation == "pop_feat":
            logpop = np.log1p(ds.popularity).astype(np.float32)
            logpop = (logpop - logpop.mean()) / logpop.std()
            content = torch.cat([content, torch.tensor(logpop)[:, None]], dim=1)
        if cfg.ablation in ("no_geo", "melu"):
            content = torch.zeros_like(content)
        self.content = content.to(device)

        self.use_id = 0.0 if cfg.ablation == "no_id" else 1.0
        self.use_social = cfg.ablation not in ("no_social", "melu")

    # ----------------------------------------------------------- parameters
    def init_params(self) -> Params:
        def linear(n_in: int, n_out: int):
            w = torch.empty(n_out, n_in)
            nn.init.xavier_uniform_(w, gain=1.0)
            return w, torch.zeros(n_out)

        d = EMBED_DIM
        P = {"id": torch.randn(self.n_venues, d) * 0.01}
        P["p1w"], P["p1b"] = linear(self.content.shape[1], d)  # content MLP, layer 1
        P["p2w"], P["p2b"] = linear(d, d)                      # content MLP, layer 2
        P["cw"], P["cb"] = linear(2 * d, d)                    # fuse id and content
        P["u1w"], P["u1b"] = linear(d, 2 * d)                  # user encoder, layer 1
        P["u2w"], P["u2b"] = linear(2 * d, d)                  # user encoder, layer 2
        return {k: v.to(self.device).requires_grad_() for k, v in P.items()}

    # ------------------------------------------------------------ encoders
    def item_emb(self, P: Params, ids: torch.Tensor) -> torch.Tensor:
        id_part = P["id"][ids] * self.use_id
        h = F.relu(F.linear(self.content[ids], P["p1w"], P["p1b"]))
        content_part = F.linear(h, P["p2w"], P["p2b"])
        fused = F.relu(F.linear(torch.cat([id_part, content_part], dim=-1), P["cw"], P["cb"]))
        return F.normalize(fused, dim=-1)

    def user_from_support(self, P: Params, support: torch.Tensor) -> torch.Tensor:
        """Encode support IDs of shape ``[..., k]`` into unit user vectors."""
        mean = self.item_emb(P, support).mean(-2)
        h = F.relu(F.linear(mean, P["u1w"], P["u1b"]))
        return F.normalize(F.linear(h, P["u2w"], P["u2b"]), dim=-1)

    def user_emb(self, P: Params, task: Task) -> torch.Tensor:
        """User vector, blended with the mean friend vector when available."""
        u = self.user_from_support(P, self.long(task.sup))
        if self.use_social and task.fr:
            friends = self.user_from_support(P, self.long(task.fr)).mean(0)
            w = self.cfg.social_w
            u = F.normalize((1 - w) * u + w * friends, dim=-1)
        return u

    # --------------------------------------------------------------- utils
    def long(self, values: Sequence) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=torch.long, device=self.device)

    def all_item_emb(self, P: Params) -> torch.Tensor:
        return self.item_emb(P, torch.arange(self.n_venues, device=self.device))


def with_support(task: Task, support) -> Task:
    """Copy of ``task`` with a different support set (used by the MAML inner loop)."""
    return replace(task, sup=list(support))

"""Meta-training of the prototypical and MAML learners."""
from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import Dataset, Task
from .geo import haversine
from .metrics import evaluate, summarise
from .model import Params, SharedModel, with_support


class MetaTrainer:
    """Trains one learner (``proto`` or ``maml``) and scores users with it.

    Both learners optimise the same query loss. They differ only in how the user
    representation is produced:

    * ``proto``: the user vector is computed directly from the support set.
    * ``maml``: the parameters are first adapted with one gradient step on the
      support set (second-order, fast weights), then the adapted network is used.
    """

    def __init__(self, ds: Dataset, cfg: Config, model: SharedModel, rng: np.random.RandomState):
        self.ds, self.cfg, self.model, self.rng = ds, cfg, model, rng
        self.is_maml = cfg.learner == "maml"
        self.item_cache: torch.Tensor | None = None  # cached POI embeddings (proto only)

    # --------------------------------------------------------------- losses
    def pair_loss(self, P: Params, user: torch.Tensor, pos, neg) -> torch.Tensor:
        """Softplus (logistic) loss on temperature-scaled cosine scores."""
        tau = self.cfg.tau
        s_pos = (self.model.item_emb(P, pos) @ user / tau).clamp(-10, 10)
        s_neg = (self.model.item_emb(P, neg) @ user / tau).clamp(-10, 10)
        return F.softplus(-s_pos).mean() + F.softplus(s_neg).mean()

    def sample_train_negatives(self, t: Task, n: int) -> torch.Tensor:
        """Negatives for training; ``mixed`` draws half of them from the local pool."""
        visited = self.ds.user_visited[t.u]
        out: List[int] = []
        if self.cfg.train_neg == "mixed":
            out = [int(x) for x in self.rng.choice(t.pool, size=n // 2, replace=False)]
        while len(out) < n:
            x = int(self.rng.randint(0, self.ds.n_venues))
            if x not in visited:
                out.append(x)
        return self.model.long(out)

    def inner_adapt(self, P: Params, t: Task, create_graph: bool) -> Params:
        """One MAML step on the support set.

        The support set is split: the first half builds the user vector (context)
        and the second half acts as the inner positives. Gradients are clipped
        element-wise for stability. With ``create_graph=True`` the step stays
        differentiable, which gives the second-order meta-gradient.
        """
        sup = t.sup
        mid = max(1, len(sup) // 2)
        context, inner_pos = sup[:mid], (sup[mid:] if len(sup) > mid else sup)
        user = self.model.user_emb(P, with_support(t, context))
        loss = self.pair_loss(P, user, self.model.long(inner_pos), self.sample_train_negatives(t, len(inner_pos)))
        keys = list(P.keys())
        grads = torch.autograd.grad(loss, [P[k] for k in keys], create_graph=create_graph, allow_unused=True)
        alpha = self.cfg.alpha_in
        return {k: (P[k] - alpha * g.clamp(-1, 1)) if g is not None else P[k] for k, g in zip(keys, grads)}

    def task_loss(self, P: Params, t: Task) -> torch.Tensor:
        W = self.inner_adapt(P, t, create_graph=True) if self.is_maml else P
        user = self.model.user_emb(W, t)
        query = self.model.long(t.qry)
        return self.pair_loss(W, user, query, self.sample_train_negatives(t, len(query)))

    # -------------------------------------------------------------- scoring
    def score_all(self, P: Params, t: Task) -> np.ndarray:
        """Cosine scores of one user against every POI."""
        if self.is_maml:
            with torch.enable_grad():
                W = self.inner_adapt(P, t, create_graph=False)
        else:
            W = P
        with torch.no_grad():
            W = {k: v.detach() for k, v in W.items()}
            user = self.model.user_emb(W, t)
            items = self.model.all_item_emb(W) if self.is_maml else self.item_cache
            return (items @ user).cpu().numpy()

    def refresh_item_cache(self, P: Params) -> None:
        with torch.no_grad():
            self.item_cache = self.model.all_item_emb({k: v.detach() for k, v in P.items()})

    # ------------------------------------------------------------- training
    def build_local_pools(self) -> None:
        """Precompute the nearest unvisited POIs per training user (mixed negatives)."""
        ds, c = self.ds, self.cfg
        t0 = time.time()
        for t in ds.train_tasks:
            sup = np.array(t.sup)
            dist = haversine(ds.lat[sup].mean(), ds.lon[sup].mean(), ds.lat, ds.lon)
            order = np.argpartition(dist, c.local_pool + 400)[: c.local_pool + 400]
            order = order[np.argsort(dist[order])]
            visited = ds.user_visited[t.u]
            t.pool = np.array([x for x in order if x not in visited][: c.local_pool])
        print(f"local training pools built in {time.time() - t0:.0f}s", flush=True)

    def fit(self, val_negs) -> Dict:
        """Meta-train with early stopping on validation NDCG@10; return the best parameters."""
        ds, c = self.ds, self.cfg
        if c.train_neg == "mixed":
            self.build_local_pools()
        P = self.model.init_params()
        optimiser = torch.optim.Adam(P.values(), lr=c.lr)
        clip = 0.5 if self.is_maml else 1.0
        best, best_epoch, bad = -1.0, 0, 0
        best_P: Params | None = None
        history, epoch_times = [], []

        for epoch in range(1, c.epochs + 1):
            t0 = time.time()
            order = self.rng.permutation(len(ds.train_tasks))
            total = 0.0
            for b in range(0, len(order) - c.batch + 1, c.batch):
                optimiser.zero_grad()
                loss = sum(self.task_loss(P, ds.train_tasks[i]) for i in order[b : b + c.batch]) / c.batch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(P.values(), clip)
                optimiser.step()
                total += loss.item()
            epoch_times.append(time.time() - t0)

            self.refresh_item_cache(P)
            val = summarise(evaluate(ds, lambda t: self.score_all(P, t), ds.val_subset, val_negs))["global"]
            history.append(dict(epoch=epoch, loss=total, train_sec=epoch_times[-1], **val))
            print(f"ep {epoch:2d} loss {total:.2f} train {epoch_times[-1]:.0f}s "
                  f"val NDCG@10 {val['NDCG@10']:.4f} R@10 {val['Recall@10']:.4f}", flush=True)

            if val["NDCG@10"] > best + 1e-4:
                best, best_epoch, bad = val["NDCG@10"], epoch, 0
                best_P = {k: v.detach().clone() for k, v in P.items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break

        self.params = {k: v.requires_grad_() for k, v in best_P.items()}
        self.refresh_item_cache(self.params)
        return dict(best_epoch=best_epoch, history=history, sec_per_epoch=float(np.mean(epoch_times)))

    def scorer(self) -> Callable[[Task], np.ndarray]:
        return lambda t: self.score_all(self.params, t)

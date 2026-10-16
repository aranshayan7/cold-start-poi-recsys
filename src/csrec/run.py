"""Command line entry point: ``python -m csrec ...``"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch

from .baselines import local_popularity_scorer, popularity_scorer
from .config import Config, parse_args
from .data import Dataset
from .metrics import evaluate, summarise
from .model import SharedModel
from .negatives import load_or_build
from .train import MetaTrainer


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    return torch.device(name)


def save(cfg: Config, payload: dict, per_user: dict) -> None:
    """Write the summary JSON and the per-user metrics used for significance tests."""
    with open(os.path.join(cfg.out, cfg.tag + ".json"), "w") as fh:
        json.dump({"tag": cfg.tag, "args": cfg.as_dict(), **payload}, fh, indent=1)
    with open(os.path.join(cfg.out, cfg.tag + "_peruser.pkl"), "wb") as fh:
        pickle.dump(per_user, fh)


def main(argv=None) -> None:
    cfg = parse_args(argv)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    print("RUN", cfg.tag, flush=True)

    ds = Dataset(cfg)
    print(ds.summary(), flush=True)
    test_g, test_l, val_g = load_or_build(ds, cfg)

    if cfg.learner in ("pop", "localpop"):
        scorer = popularity_scorer(ds, rng) if cfg.learner == "pop" else local_popularity_scorer(ds)
        per_user = evaluate(ds, scorer, ds.test_tasks, test_g, test_l, full=True)
        summary = summarise(per_user)
        print(json.dumps(summary, indent=1))
        save(cfg, {"test": summary}, per_user)
        return

    model = SharedModel(ds, cfg, resolve_device(cfg.device))
    trainer = MetaTrainer(ds, cfg, model, rng)
    info = trainer.fit(val_g)

    t0 = time.time()
    per_user = evaluate(ds, trainer.scorer(), ds.test_tasks, test_g, test_l, full=True)
    eval_seconds = time.time() - t0
    summary = summarise(per_user)
    print(json.dumps(summary, indent=1))
    save(cfg, {"best_epoch": info["best_epoch"], "sec_per_epoch": info["sec_per_epoch"],
               "eval_sec_per_user": eval_seconds / len(ds.test_tasks),
               "history": info["history"], "test": summary}, per_user)


if __name__ == "__main__":
    main()

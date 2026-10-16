"""Experiment configuration and command line interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields

LEARNERS = ("proto", "maml", "pop", "localpop")
ABLATIONS = ("none", "no_geo", "no_social", "no_id", "pop_feat", "melu")
TRAIN_NEGATIVES = ("uniform", "mixed")


@dataclass
class Config:
    """All hyper-parameters and switches of a single run.

    Attributes
    ----------
    data : str
        Directory with ``checkins.parquet``, ``pois.parquet``, ``splits.pkl`` and
        ``social_adj.pkl`` (see ``data/README.md``).
    out : str
        Directory where result files are written.
    learner : str
        ``proto`` (prototypical network), ``maml`` (second-order MAML),
        ``pop`` (global popularity) or ``localpop`` (popularity discounted by
        distance to the support centroid).
    ablation : str
        Variant of the shared representation: ``none`` (full model), ``no_geo``
        (coordinates and category zeroed), ``no_social`` (no friend blending),
        ``no_id`` (no learned POI embedding), ``pop_feat`` (adds a log
        popularity feature) or ``melu`` (content only, no social signal).
    k : int
        Number of support POIs per user used at training and test time.
    k_max : int
        The query set starts after this many unique POIs, so the query is
        identical for every value of ``k``.
    q : int
        Query set size, identical for every model.
    """

    data: str = "data/sample"
    out: str = "results"
    learner: str = "proto"
    ablation: str = "none"
    k: int = 10
    k_max: int = 10
    q: int = 20
    seed: int = 1
    epochs: int = 15
    patience: int = 3
    batch: int = 16
    lr: float = 1e-3
    alpha_in: float = 1e-3
    tau: float = 0.05
    social_w: float = 0.25
    n_friends: int = 3
    neg_per_pos: int = 100
    local_pool: int = 5000
    n_val: int = 400
    train_neg: str = "uniform"
    threads: int = 2
    device: str = "cpu"

    @property
    def tag(self) -> str:
        """Unique run identifier used for all output file names."""
        tag = f"{self.learner}_{self.ablation}_k{self.k}_s{self.seed}"
        if self.learner == "maml":
            tag += f"_a{self.alpha_in:g}"
        if self.train_neg == "mixed":
            tag += "_mixneg"
        return tag

    def as_dict(self) -> dict:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser from the dataclass defaults."""
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog="csrec",
        description="Train and evaluate a cold-start POI recommender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    helps = {
        "data": "directory with the processed data files",
        "out": "directory for result files",
        "k": "support size at training and test time",
        "k_max": "query starts after this many unique POIs",
        "q": "query size (shared by all models)",
        "alpha_in": "MAML inner-loop learning rate",
        "social_w": "weight of the friend prototype in the blended user vector",
        "n_friends": "maximum number of friends used per user",
        "neg_per_pos": "evaluation negatives per query positive",
        "local_pool": "size of the nearest-unvisited pool for local negatives",
        "n_val": "number of validation users used for early stopping",
        "threads": "number of CPU threads for PyTorch",
        "device": "cpu, cuda or mps",
    }
    choices = {"learner": LEARNERS, "ablation": ABLATIONS, "train_neg": TRAIN_NEGATIVES}
    for f in fields(Config):
        kwargs = {"default": getattr(defaults, f.name), "help": helps.get(f.name, " ")}
        if f.name in choices:
            kwargs["choices"] = choices[f.name]
        else:
            kwargs["type"] = type(getattr(defaults, f.name))
        parser.add_argument(f"--{f.name}", **kwargs)
    return parser


def parse_args(argv=None) -> Config:
    ns = build_parser().parse_args(argv)
    return Config(**vars(ns))

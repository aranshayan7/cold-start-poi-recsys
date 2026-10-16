#!/usr/bin/env python3
"""Aggregate run outputs into mean/std tables and paired significance tests.

Reads every ``<tag>.json`` produced by ``python -m csrec`` from ``--results`` and
writes ``summary.json``, ``summary.md`` and ``wilcoxon.json`` next to them.
"""
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20"]
SETTINGS = ["global", "local", "full"]
# (system A, system B) pairs compared with a paired Wilcoxon test over test users.
PAIRS = [
    ("proto_none_k10", "maml_none_k10_a0.001"),
    ("proto_none_k10", "pop_none_k10"),
    ("maml_none_k10_a0.001", "pop_none_k10"),
    ("proto_none_k10", "proto_no_social_k10"),
    ("proto_none_k10", "localpop_none_k10"),
    ("proto_pop_feat_k10", "pop_none_k10"),
    ("proto_pop_feat_k10", "proto_none_k10"),
    ("proto_pop_feat_k10_mixneg", "pop_none_k10"),
    ("proto_pop_feat_k10_mixneg", "proto_pop_feat_k10"),
]


def load_runs(folder):
    """Group run JSONs by (learner, variant, k) and seed."""
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(folder, "*.json")):
        if os.path.basename(path) in ("summary.json", "wilcoxon.json"):
            continue
        run = json.load(open(path))
        a = run["args"]
        variant = a["ablation"] + ("+mixneg" if a.get("train_neg") == "mixed" else "")
        runs[(a["learner"], variant, a["k"])][a["seed"]] = run
    return runs


def summarise(runs):
    rows = []
    for key in sorted(runs):
        seeds = sorted(runs[key])
        for setting in SETTINGS:
            vals = {m: [runs[key][s]["test"][setting][m] for s in seeds if setting in runs[key][s]["test"]] for m in METRICS}
            if not vals["NDCG@10"]:
                continue
            rows.append(dict(
                learner=key[0], variant=key[1], k=key[2], setting=setting, n_seeds=len(seeds),
                **{m: float(np.mean(v)) for m, v in vals.items()},
                **{m + "_std": float(np.std(v)) for m, v in vals.items()},
            ))
    return rows


def cost_table(runs):
    lines = ["| learner | variant | k | sec/epoch | eval sec/user | best epochs |", "|---|---|---|---|---|---|"]
    for key in sorted(runs):
        seeds = sorted(runs[key])
        first = runs[key][seeds[0]]
        if "sec_per_epoch" not in first:
            continue
        lines.append("| {} | {} | {} | {:.1f} | {:.4f} | {} |".format(
            key[0], key[1], key[2],
            np.mean([runs[key][s]["sec_per_epoch"] for s in seeds]),
            np.mean([runs[key][s]["eval_sec_per_user"] for s in seeds]),
            [runs[key][s]["best_epoch"] for s in seeds]))
    return "\n".join(lines)


def markdown(rows):
    lines = []
    for setting in SETTINGS:
        sub = [r for r in rows if r["setting"] == setting]
        if not sub:
            continue
        lines += [f"### Candidate set: {setting}", "",
                  "| learner | variant | k | seeds | " + " | ".join(METRICS) + " |",
                  "|---|---|---|---|" + "---|" * len(METRICS)]
        for r in sub:
            cells = " | ".join(f"{r[m]:.4f} ± {r[m + '_std']:.4f}" for m in METRICS)
            lines.append(f"| {r['learner']} | {r['variant']} | {r['k']} | {r['n_seeds']} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def per_user(folder, tag, setting, metric="NDCG@10"):
    path = os.path.join(folder, f"{tag}_peruser.pkl")
    if not os.path.exists(path):
        return None
    return np.array([x[metric] for x in pickle.load(open(path, "rb"))[setting]])


def significance(folder, seed):
    tests = []
    for a, b in PAIRS:
        tag_a, tag_b = _with_seed(a, seed), _with_seed(b, seed)
        for setting in SETTINGS:
            x, y = per_user(folder, tag_a, setting), per_user(folder, tag_b, setting)
            if x is None or y is None or len(x) != len(y):
                continue
            tests.append(dict(a=tag_a, b=tag_b, setting=setting, mean_a=float(x.mean()), mean_b=float(y.mean()),
                              p=float(wilcoxon(x, y).pvalue)))
    return tests


def _with_seed(name, seed):
    """Insert ``_s<seed>`` after the ``_k<k>`` token of a tag stem."""
    parts = name.split("_")
    i = next(j for j, p in enumerate(parts) if p.startswith("k") and p[1:].isdigit())
    return "_".join(parts[: i + 1] + [f"s{seed}"] + parts[i + 1 :])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=1, help="seed used for the paired tests")
    a = ap.parse_args()

    runs = load_runs(a.results)
    if not runs:
        raise SystemExit(f"no result files found in {a.results}")
    rows = summarise(runs)
    tests = significance(a.results, a.seed)

    json.dump(rows, open(os.path.join(a.results, "summary.json"), "w"), indent=1)
    json.dump(tests, open(os.path.join(a.results, "wilcoxon.json"), "w"), indent=1)
    text = markdown(rows) + "\n### Training and scoring cost\n\n" + cost_table(runs) + "\n"
    if tests:
        text += "\n### Paired Wilcoxon tests (NDCG@10, per-user)\n\n| A | B | candidate set | mean A | mean B | p |\n|---|---|---|---|---|---|\n"
        text += "\n".join(f"| {t['a']} | {t['b']} | {t['setting']} | {t['mean_a']:.4f} | {t['mean_b']:.4f} | {t['p']:.2e} |" for t in tests) + "\n"
    open(os.path.join(a.results, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()

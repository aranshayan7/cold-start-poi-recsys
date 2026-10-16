# import torch
# import numpy as np
# from collections import defaultdict
#
#
# def compute_metrics(model, tasks_batch, device, top_k_list=[10, 20], max_neg=100):
#     """ارزیابی استاندارد برای Prototypical Networks"""
#     model.eval()
#     all_metrics = defaultdict(list)
#
#     with torch.no_grad():
#         for i in range(len(tasks_batch['user_idx'])):
#             support = tasks_batch['support_venues'][i].to(device)
#             query = tasks_batch['query_venues'][i].to(device)
#             social = tasks_batch['social_neighbors'][i].to(device) if len(tasks_batch['social_neighbors'][i]) > 0 else None
#
#             if len(support) < 3 or len(query) == 0:
#                 continue
#
#             # ساخت Prototype
#             prototype = model.compute_prototype(support, social)
#
#             # Sampled Ranking
#             true_pos = set(query.cpu().numpy())
#             support_set = set(support.cpu().numpy())
#
#             n_neg = min(len(true_pos) * 10, max_neg)
#             neg_samples = []
#             attempts = 0
#             while len(neg_samples) < n_neg and attempts < n_neg * 5:
#                 neg = np.random.randint(0, model.n_venues)   # حالا n_venues وجود دارد
#                 if neg not in true_pos and neg not in support_set:
#                     neg_samples.append(neg)
#                 attempts += 1
#
#             candidates = list(true_pos) + neg_samples
#             cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
#
#             scores = model.get_poi_embedding(cand_tensor) @ prototype
#
#             for k in top_k_list:
#                 topk_idx = torch.topk(scores, k=min(k, len(candidates))).indices
#                 topk_venues = [candidates[idx] for idx in topk_idx]
#
#                 hits = len(set(topk_venues) & true_pos)
#
#                 precision = hits / k
#                 recall = hits / len(true_pos) if true_pos else 0.0
#                 ndcg = compute_ndcg(topk_venues, true_pos, k)
#
#                 all_metrics[f'Precision@{k}'].append(precision)
#                 all_metrics[f'Recall@{k}'].append(recall)
#                 all_metrics[f'NDCG@{k}'].append(ndcg)
#
#     return {k: np.mean(v) for k, v in all_metrics.items()}
#
#
# def compute_ndcg(ranked_list, true_positives, k):
#     dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in true_positives)
#     idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_positives), k)))
#     return dcg / idcg if idcg > 0 else 0.0

import torch
import numpy as np
from collections import defaultdict


def compute_metrics(model, tasks_batch, device, top_k_list=[10, 20]):
    """
    ارزیابی استاندارد - همون protocol مدل 1
    100 negative per positive (نه 10)
    همه batch های val (نه یک batch)
    """
    model.eval()
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for i in range(len(tasks_batch['user_idx'])):
            support = tasks_batch['support_venues'][i].to(device)
            query = tasks_batch['query_venues'][i].to(device)
            sn = tasks_batch['social_neighbors'][i]
            social = sn.to(device) if len(sn) > 0 else None

            if len(support) < 3 or len(query) == 0:
                continue

            prototype = model.compute_prototype(support, social)

            true_pos = set(query.cpu().numpy().tolist())
            support_set = set(support.cpu().numpy().tolist())
            excluded = true_pos | support_set

            # 100 negatives per positive - same as model 1
            n_neg_samples = min(len(true_pos) * 100, 2000)
            neg_samples = []
            attempts = 0
            visited = set(excluded)
            while len(neg_samples) < n_neg_samples and attempts < n_neg_samples * 5:
                neg = np.random.randint(0, model.n_venues)
                if neg not in visited:
                    neg_samples.append(neg)
                    visited.add(neg)
                attempts += 1

            candidates = list(true_pos) + neg_samples
            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.get_poi_embedding(cand_tensor) @ prototype

            for k in top_k_list:
                topk_idx = torch.topk(scores, k=min(k, len(candidates))).indices
                topk_venues = [candidates[idx] for idx in topk_idx.cpu().numpy()]

                hits = len(set(topk_venues) & true_pos)
                precision = hits / k
                recall = hits / len(true_pos) if true_pos else 0.0
                ndcg = compute_ndcg(topk_venues, true_pos, k)

                all_metrics[f'Precision@{k}'].append(precision)
                all_metrics[f'Recall@{k}'].append(recall)
                all_metrics[f'NDCG@{k}'].append(ndcg)

    return {k: np.mean(v) for k, v in all_metrics.items()}


def compute_ndcg(ranked_list, true_positives, k):
    dcg = sum(1.0 / np.log2(i + 2)
              for i, item in enumerate(ranked_list[:k])
              if item in true_positives)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(true_positives), k)))
    return dcg / idcg if idcg > 0 else 0.0
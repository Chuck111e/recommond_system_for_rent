"""Offline ranking metrics with a time-based holdout."""

from __future__ import annotations

import math

import pandas as pd


def evaluate_recommendations(pipeline, test: pd.DataFrame, k: int = 10) -> dict[str, float]:
    target = test.groupby("user_id")["item_id"].apply(set).to_dict()
    hits = []
    ndcgs = []
    recalls = []
    recommended = set()
    for user_id, truth in target.items():
        result = pipeline.recommend(user_id, k=k)
        predicted = result["item_id"].tolist() if not result.empty else []
        recommended.update(predicted)
        recalls.append(len(set(predicted) & truth) / max(1, len(truth)))
        hit_positions = [idx + 1 for idx, item_id in enumerate(predicted) if item_id in truth]
        hits.append(float(bool(hit_positions)))
        ndcgs.append(sum(1 / math.log2(position + 1) for position in hit_positions))
    user_count = max(1, len(target))
    return {
        f"recall@{k}": float(sum(recalls) / user_count),
        f"hit_rate@{k}": float(sum(hits) / user_count),
        f"ndcg@{k}": float(sum(ndcgs) / user_count),
        "item_coverage": float(len(recommended) / max(1, len(pipeline.items))),
        "evaluated_users": float(len(target)),
    }

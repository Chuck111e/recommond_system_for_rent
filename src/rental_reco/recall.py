"""Candidate generation: popularity, item-based CF and content recall."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class PopularityRecall:
    name = "popularity"

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "PopularityRecall":
        counts = train.loc[train["label"] == 1].groupby("item_id").size()
        self.scores = counts.reindex(items["item_id"], fill_value=0).astype(float).to_dict()
        self.items = set(items["item_id"])
        return self

    def retrieve(self, user_id: str, seen: set[str], k: int) -> list[tuple[str, float]]:
        candidates = [(item, score) for item, score in self.scores.items() if item not in seen]
        return sorted(candidates, key=lambda x: (-x[1], x[0]))[:k]


class ItemCFRecall:
    name = "itemcf"

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "ItemCFRecall":
        positive = train[train["label"] == 1]
        self.user_items = positive.groupby("user_id")["item_id"].apply(set).to_dict()
        item_users: dict[str, set[str]] = defaultdict(set)
        for user_id, item_ids in self.user_items.items():
            for item_id in item_ids:
                item_users[item_id].add(user_id)
        self.similarity: dict[str, dict[str, float]] = defaultdict(dict)
        for item_a, users_a in item_users.items():
            for item_b, users_b in item_users.items():
                if item_a == item_b:
                    continue
                overlap = len(users_a & users_b)
                if overlap:
                    self.similarity[item_a][item_b] = overlap / np.sqrt(len(users_a) * len(users_b))
        self.items = set(items["item_id"])
        return self

    def retrieve(self, user_id: str, seen: set[str], k: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for history_item in self.user_items.get(user_id, set()):
            for candidate, similarity in self.similarity.get(history_item, {}).items():
                if candidate not in seen:
                    scores[candidate] += similarity
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


class ContentRecall:
    name = "content"

    def fit(self, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> "ContentRecall":
        self.users = users.set_index("user_id")
        self.items = items.set_index("item_id")
        self.history = train[train["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def retrieve(self, user_id: str, seen: set[str], k: int) -> list[tuple[str, float]]:
        user = self.users.loc[user_id] if user_id in self.users.index else None
        if user is None:
            return []
        preferred = set(str(user["preferred_categories"]).split("|"))
        rows = []
        for item_id, item in self.items.iterrows():
            if item_id in seen:
                continue
            score = float(item["category"] in preferred) * 2.0
            score += float(item["city"] == user["city"]) * 0.4
            score += float(item["quality_score"]) * 0.2
            rows.append((item_id, score))
        return sorted(rows, key=lambda x: (-x[1], x[0]))[:k]


class MatrixFactorRecall:
    """Implicit-feedback matrix factorization recall for warm users."""

    name = "matrix_factor"

    def fit(self, train: pd.DataFrame, items: pd.DataFrame) -> "MatrixFactorRecall":
        positive = train[train["label"] == 1]
        self.item_ids = items["item_id"].tolist()
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        user_ids = sorted(positive["user_id"].unique())
        self.user_index = {user_id: index for index, user_id in enumerate(user_ids)}
        matrix = np.zeros((len(user_ids), len(self.item_ids)), dtype=np.float32)
        for row in positive.itertuples(index=False):
            if row.user_id in self.user_index and row.item_id in self.item_index:
                matrix[self.user_index[row.user_id], self.item_index[row.item_id]] = 1.0
        self.user_latent = np.zeros((len(user_ids), 1), dtype=np.float32)
        self.item_latent = np.zeros((len(self.item_ids), 1), dtype=np.float32)
        if matrix.shape[0] > 1 and matrix.shape[1] > 1 and matrix.sum() > 0:
            components = max(1, min(16, matrix.shape[0] - 1, matrix.shape[1] - 1))
            model = TruncatedSVD(n_components=components, random_state=42)
            self.user_latent = model.fit_transform(matrix)
            self.item_latent = model.components_.T
        return self

    def retrieve(self, user_id: str, seen: set[str], k: int) -> list[tuple[str, float]]:
        if user_id not in self.user_index:
            return []
        scores = self.user_latent[self.user_index[user_id]] @ self.item_latent.T
        rows = [(item_id, float(scores[index])) for index, item_id in enumerate(self.item_ids) if item_id not in seen]
        return sorted(rows, key=lambda x: (-x[1], x[0]))[:k]


class HybridRetriever:
    """Fuse multiple recall channels and retain their explanations."""

    def __init__(self, recalls: list[tuple[object, float]] | None = None):
        self.recalls = recalls or []

    def fit(self, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> "HybridRetriever":
        if not self.recalls:
            self.recalls = [
                (PopularityRecall(), 0.25),
                (ItemCFRecall(), 0.35),
                (MatrixFactorRecall(), 0.20),
                (ContentRecall(), 0.20),
            ]
        for recall, _ in self.recalls:
            if isinstance(recall, ContentRecall):
                recall.fit(train, users, items)
            else:
                recall.fit(train, items)
        return self

    def retrieve(self, user_id: str, train: pd.DataFrame, k: int = 80) -> pd.DataFrame:
        seen = set(train.loc[(train["user_id"] == user_id) & (train["label"] == 1), "item_id"])
        fused: dict[str, dict[str, object]] = {}
        per_channel_k = max(k, 20)
        for recall, weight in self.recalls:
            for rank, (item_id, score) in enumerate(recall.retrieve(user_id, seen, per_channel_k), start=1):
                entry = fused.setdefault(item_id, {"item_id": item_id, "recall_score": 0.0, "recall_sources": []})
                entry["recall_score"] = float(entry["recall_score"]) + float(weight) * float(score) / rank
                entry["recall_sources"].append(recall.name)
        rows = sorted(fused.values(), key=lambda x: (-float(x["recall_score"]), str(x["item_id"])))[:k]
        return pd.DataFrame(rows, columns=["item_id", "recall_score", "recall_sources"])

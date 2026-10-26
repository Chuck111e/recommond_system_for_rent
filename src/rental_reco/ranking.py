"""Learning-to-rank baseline using a calibrated logistic model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .features import FeatureBuilder


class LogisticRanker:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(solver="liblinear", max_iter=500, class_weight="balanced", random_state=seed)
        self.fitted = False

    def fit(
        self,
        train: pd.DataFrame,
        users: pd.DataFrame,
        items: pd.DataFrame,
        negative_ratio: int = 2,
    ) -> "LogisticRanker":
        rng = np.random.default_rng(self.seed)
        positive = train[train["label"] == 1][["user_id", "item_id"]].drop_duplicates()
        if positive.empty:
            raise ValueError("训练数据至少需要一条 label=1 的正反馈")
        item_ids = items["item_id"].to_numpy()
        seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        negatives = []
        for row in positive.itertuples(index=False):
            available = np.array([item for item in item_ids if item not in seen.get(row.user_id, set())])
            if len(available) == 0:
                continue
            for item_id in rng.choice(available, size=min(negative_ratio, len(available)), replace=False):
                negatives.append({"user_id": row.user_id, "item_id": item_id})
        if not negatives:
            raise ValueError("训练数据没有可采样的负样本，请增加商品数量或减少每个用户的历史行为")
        samples = pd.concat([positive.assign(label=1), pd.DataFrame(negatives).assign(label=0)], ignore_index=True)
        builder = FeatureBuilder(users, items)
        matrix = self._prepare_matrix(self.vectorizer.fit_transform(builder.to_dicts(samples)))
        self.model.fit(matrix, samples["label"].to_numpy())
        self.fitted = True
        return self

    def predict(self, pairs: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ranker has not been fitted")
        builder = FeatureBuilder(users, items)
        matrix = self._prepare_matrix(self.vectorizer.transform(builder.to_dicts(pairs)))
        return self.model.predict_proba(matrix)[:, 1]

    @staticmethod
    def _prepare_matrix(matrix):
        """Keep sparse indices compatible with recent scikit-learn releases."""

        if hasattr(matrix, "indices"):
            matrix = matrix.tocsr(copy=True)
            matrix.indices = matrix.indices.astype("int32", copy=False)
            matrix.indptr = matrix.indptr.astype("int32", copy=False)
        return matrix

    def rank(self, candidates: pd.DataFrame, user_id: str, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            result = candidates.copy()
            result["user_id"] = pd.Series(dtype=str)
            result["rank_score"] = pd.Series(dtype=float)
            return result
        pairs = candidates.copy()
        pairs.insert(0, "user_id", user_id)
        pairs["rank_score"] = self.predict(pairs[["user_id", "item_id", "recall_score"]], users, items)
        return pairs.sort_values(["rank_score", "recall_score", "item_id"], ascending=[False, False, True]).reset_index(drop=True)

"""End-to-end training and recommendation pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data import generate_rental_data, load_csv_bundle, temporal_split
from .evaluation import evaluate_recommendations
from .ranking import LogisticRanker
from .recall import HybridRetriever


class RecommendationPipeline:
    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        self.users = pd.DataFrame()
        self.items = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test = pd.DataFrame()
        self.retriever = HybridRetriever()
        self.ranker = LogisticRanker(self.config.seed)
        self.metrics: dict[str, float] = {}

    def train(self, users: pd.DataFrame | None = None, items: pd.DataFrame | None = None, interactions: pd.DataFrame | None = None) -> "RecommendationPipeline":
        if users is None or items is None or interactions is None:
            users, items, interactions = generate_rental_data(self.config)
        self.users, self.items = users.copy(), items.copy()
        self.train_data, self.test = temporal_split(interactions, self.config.test_size)
        self.retriever.fit(self.train_data, self.users, self.items)
        self.ranker.fit(self.train_data, self.users, self.items, self.config.negative_ratio)
        self.metrics = evaluate_recommendations(self, self.test, self.config.top_k)
        return self

    def train_from_csv(self, data_dir: str | Path) -> "RecommendationPipeline":
        return self.train(*load_csv_bundle(data_dir))

    def recommend(self, user_id: str, k: int | None = None) -> pd.DataFrame:
        self._ensure_trained()
        k = k or self.config.top_k
        candidates = self.retriever.retrieve(user_id, self.train_data, max(k, self.config.candidate_k))
        ranked = self.ranker.rank(candidates, user_id, self.users, self.items)
        result = ranked.head(k).merge(self.items, on="item_id", how="left", suffixes=("", "_item"))
        if not result.empty:
            result["reason"] = result["recall_sources"].apply(lambda _: ";".join(_) if isinstance(_, list) else "混合召回")
        else:
            result["reason"] = pd.Series(dtype=str)
        return result

    def save_metrics(self, path: str | Path | None = None) -> Path:
        destination = Path(path or self.config.artifact_dir / "metrics.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    def save(self, path: str | Path | None = None) -> Path:
        destination = Path(path or self.config.artifact_dir / "pipeline.pkl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RecommendationPipeline":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a RecommendationPipeline")
        return model

    def _ensure_trained(self) -> None:
        if self.users.empty or self.items.empty or self.train_data.empty:
            raise RuntimeError("pipeline is not trained; call train() first")

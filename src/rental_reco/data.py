"""Deterministic, privacy-safe data generation and temporal splitting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import ProjectConfig


CATEGORIES = ("手机", "相机", "电脑", "平板", "无人机", "游戏机")
BRANDS = ("Apple", "Samsung", "华为", "小米", "Sony", "DJI", "联想")
CITIES = ("深圳", "广州", "东莞", "杭州", "上海", "北京", "成都")
CONDITIONS = ("全新", "95新", "9成新")


def load_csv_bundle(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load users.csv, items.csv and interactions.csv using the project contract."""

    root = Path(data_dir)
    users = pd.read_csv(root / "users.csv")
    items = pd.read_csv(root / "items.csv")
    interactions = pd.read_csv(root / "interactions.csv", parse_dates=["timestamp"])
    required = {
        "users": {"user_id", "city", "age_bucket", "gender", "budget_level", "preferred_categories"},
        "items": {"item_id", "category", "brand", "city", "condition", "daily_rent", "quality_score"},
        "interactions": {"user_id", "item_id", "event_type", "label", "timestamp"},
    }
    for name, frame in (("users", users), ("items", items), ("interactions", interactions)):
        missing = required[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv 缺少字段: {sorted(missing)}")
    return users, items, interactions


def save_csv_bundle(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame, data_dir: str | Path) -> Path:
    """Persist a generated bundle so the same data can be inspected or replayed."""

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    users.to_csv(root / "users.csv", index=False)
    items.to_csv(root / "items.csv", index=False)
    interactions.to_csv(root / "interactions.csv", index=False)
    return root


def generate_rental_data(config: ProjectConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a realistic rental domain sample without company data.

    The generator creates user profiles, a product catalog, and explicit/implicit
    behavior events. A fixed seed makes every demo and test reproducible.
    """

    config = config or ProjectConfig()
    rng = np.random.default_rng(config.seed)
    users = _generate_users(config, rng)
    items = _generate_items(config, rng)
    interactions = _generate_interactions(config, rng, users, items)
    return users, items, interactions


def _generate_users(config: ProjectConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for idx in range(config.n_users):
        preference = rng.choice(CATEGORIES, size=2, replace=False)
        rows.append(
            {
                "user_id": f"u{idx + 1:04d}",
                "city": CITIES[idx % len(CITIES)],
                "age_bucket": int(rng.integers(1, 6)),
                "gender": rng.choice(("男", "女", "未知"), p=(0.47, 0.47, 0.06)),
                "budget_level": int(rng.integers(1, 6)),
                "preferred_categories": "|".join(preference),
            }
        )
    return pd.DataFrame(rows)


def _generate_items(config: ProjectConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for idx in range(config.n_items):
        category = CATEGORIES[idx % len(CATEGORIES)]
        base_rent = {"手机": 18, "相机": 55, "电脑": 42, "平板": 25, "无人机": 80, "游戏机": 35}[category]
        rows.append(
            {
                "item_id": f"i{idx + 1:04d}",
                "category": category,
                "brand": BRANDS[idx % len(BRANDS)],
                "city": CITIES[(idx * 3) % len(CITIES)],
                "condition": CONDITIONS[idx % len(CONDITIONS)],
                "daily_rent": round(max(8, base_rent + rng.normal(0, base_rent * 0.16)), 2),
                "quality_score": round(float(np.clip(rng.normal(0.78, 0.1), 0.45, 0.99)), 4),
            }
        )
    return pd.DataFrame(rows)


def _generate_interactions(
    config: ProjectConfig,
    rng: np.random.Generator,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(days=config.history_days)
    popularity = rng.normal(0, 0.25, len(items))
    item_matrix = items.reset_index(drop=True)
    rows = []
    for user in users.itertuples(index=False):
        preferred = set(str(user.preferred_categories).split("|"))
        utility = popularity.copy()
        utility += item_matrix["category"].isin(preferred).to_numpy() * 1.35
        utility += (item_matrix["city"].to_numpy() == user.city) * 0.35
        budget_distance = np.abs(item_matrix["daily_rent"].to_numpy() - user.budget_level * 20)
        utility -= budget_distance / 120
        probability = np.exp(utility - utility.max())
        probability /= probability.sum()
        chosen = rng.choice(len(item_matrix), size=min(config.interactions_per_user, len(item_matrix)), replace=False, p=probability)
        for position, item_idx in enumerate(chosen):
            event_type = rng.choice(("view", "favorite", "order"), p=(0.53, 0.22, 0.25))
            label = int(event_type in ("favorite", "order"))
            event_time = start + timedelta(days=int(rng.integers(0, config.history_days)), minutes=int(rng.integers(0, 1440)))
            rows.append(
                {
                    "user_id": user.user_id,
                    "item_id": item_matrix.iloc[item_idx].item_id,
                    "event_type": event_type,
                    "label": label,
                    "timestamp": event_time,
                    "position": position,
                }
            )
    result = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return result


def temporal_split(interactions: pd.DataFrame, test_size: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's positive events by time to avoid future leakage."""

    positive = interactions[interactions["label"] == 1].copy()
    positive["timestamp"] = pd.to_datetime(positive["timestamp"], utc=True)
    test_size = max(1, int(test_size))
    test_indices = positive.sort_values("timestamp").groupby("user_id", sort=False).tail(test_size).index
    test = positive.loc[test_indices].sort_values("timestamp").reset_index(drop=True)
    train = interactions.drop(index=test_indices).sort_values("timestamp").reset_index(drop=True)
    return train, test

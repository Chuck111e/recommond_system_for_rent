"""Feature construction shared by training and online ranking."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FeatureBuilder:
    def __init__(self, users: pd.DataFrame, items: pd.DataFrame):
        self.users = users.set_index("user_id")
        self.items = items.set_index("item_id")

    def pair_frame(self, pairs: pd.DataFrame) -> pd.DataFrame:
        result = pairs.copy()
        user_features = self.users.rename(columns={column: f"{column}_user" for column in self.users.columns})
        item_features = self.items.rename(columns={column: f"{column}_item" for column in self.items.columns})
        result = result.join(user_features, on="user_id")
        result = result.join(item_features, on="item_id")
        return result

    def to_dicts(self, pairs: pd.DataFrame) -> list[dict[str, Any]]:
        frame = self.pair_frame(pairs)
        return [self._row_to_dict(row) for _, row in frame.iterrows()]

    @staticmethod
    def _row_to_dict(row: pd.Series) -> dict[str, Any]:
        def number(value: Any) -> float:
            if value is None or pd.isna(value):
                return 0.0
            return float(value)

        categories = set(str(row.get("preferred_categories_user", "")).split("|"))
        city_match = float(row.get("city_user", "") == row.get("city_item", ""))
        category_match = float(row.get("category_item", "") in categories)
        budget = number(row.get("budget_level_user", 0)) * 20
        rent = number(row.get("daily_rent_item", 0))
        feature: dict[str, Any] = {
            "user_city=" + str(row.get("city_user", "未知")): 1.0,
            "item_city=" + str(row.get("city_item", "未知")): 1.0,
            "gender=" + str(row.get("gender_user", "未知")): 1.0,
            "category=" + str(row.get("category_item", "未知")): 1.0,
            "brand=" + str(row.get("brand_item", "未知")): 1.0,
            "condition=" + str(row.get("condition_item", "未知")): 1.0,
            "age_bucket=" + str(row.get("age_bucket_user", "未知")): 1.0,
            "budget_level=" + str(row.get("budget_level_user", "未知")): 1.0,
            "city_match": city_match,
            "category_match": category_match,
            "rent_gap": abs(rent - budget),
            "daily_rent": rent,
            "quality_score": number(row.get("quality_score_item", 0)),
            "recall_score": number(row.get("recall_score", 0)),
        }
        return feature

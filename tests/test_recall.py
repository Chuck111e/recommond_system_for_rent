import pandas as pd

from rental_reco.config import ProjectConfig
from rental_reco.data import generate_rental_data, temporal_split
from rental_reco.recall import HybridRetriever


def test_temporal_split_keeps_holdout_after_training_data():
    _, _, interactions = generate_rental_data(ProjectConfig(n_users=12, n_items=30, seed=7))
    train, test = temporal_split(interactions, test_size=1)
    for user_id, rows in test.groupby("user_id"):
        train_positive = train.loc[(train["user_id"] == user_id) & (train["label"] == 1), "timestamp"]
        if not train_positive.empty:
            assert rows["timestamp"].min() >= train_positive.max()


def test_hybrid_recall_excludes_positive_history():
    users, items, interactions = generate_rental_data(ProjectConfig(n_users=10, n_items=25, seed=3))
    train, _ = temporal_split(interactions, test_size=1)
    retriever = HybridRetriever().fit(train, users, items)
    user_id = users.iloc[0]["user_id"]
    result = retriever.retrieve(user_id, train, k=10)
    seen = set(train.loc[(train.user_id == user_id) & (train.label == 1), "item_id"])
    assert not seen.intersection(result.item_id)

from rental_reco.config import ProjectConfig
from rental_reco.pipeline import RecommendationPipeline


def test_pipeline_trains_and_returns_ranked_items(tmp_path):
    config = ProjectConfig(n_users=18, n_items=36, interactions_per_user=10, top_k=5, candidate_k=20, artifact_dir=tmp_path)
    pipeline = RecommendationPipeline(config).train()
    user_id = pipeline.users.iloc[0]["user_id"]
    result = pipeline.recommend(user_id, k=5)
    assert len(result) <= 5
    assert {"item_id", "rank_score", "reason"}.issubset(result.columns)
    assert pipeline.metrics["evaluated_users"] > 0
    assert pipeline.save_metrics().exists()

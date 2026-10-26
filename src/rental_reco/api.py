from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .pipeline import RecommendationPipeline

app = FastAPI(title="Rental Recommendation API", version="0.1.0")
pipeline: RecommendationPipeline | None = None


def get_pipeline() -> RecommendationPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RecommendationPipeline().train()
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, float]:
    return get_pipeline().metrics


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(default=10, ge=1, le=100)) -> dict:
    model = get_pipeline()
    if user_id not in set(model.users["user_id"]):
        raise HTTPException(status_code=404, detail="unknown user_id")
    result = model.recommend(user_id, k=k)
    return {"user_id": user_id, "items": result.to_dict("records")}

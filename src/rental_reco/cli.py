from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectConfig
from .data import generate_rental_data, save_csv_bundle
from .pipeline import RecommendationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rental product recommendation demo")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="train, evaluate and print recommendations")
    demo.add_argument("--users", type=int, default=240)
    demo.add_argument("--items", type=int, default=120)
    demo.add_argument("--top-k", type=int, default=10)
    demo.add_argument("--artifact-dir", default="artifacts")
    demo.add_argument("--data-dir", default=None, help="optional directory containing users.csv/items.csv/interactions.csv")
    serve = sub.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        config = ProjectConfig(n_users=args.users, n_items=args.items, top_k=args.top_k, artifact_dir=Path(args.artifact_dir))
        pipeline = RecommendationPipeline(config)
        if args.data_dir:
            pipeline = pipeline.train_from_csv(args.data_dir)
        else:
            users, items, interactions = generate_rental_data(config)
            save_csv_bundle(users, items, interactions, config.artifact_dir / "data")
            pipeline = pipeline.train(users, items, interactions)
        pipeline.save_metrics()
        pipeline.save()
        user_id = pipeline.users.iloc[0]["user_id"]
        print(json.dumps({"metrics": pipeline.metrics, "user_id": user_id, "recommendations": pipeline.recommend(user_id).to_dict("records")}, ensure_ascii=False, indent=2, default=str))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("rental_reco.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

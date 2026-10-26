from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration shared by data generation, training and serving."""

    seed: int = 42
    n_users: int = 240
    n_items: int = 120
    interactions_per_user: int = 18
    history_days: int = 60
    test_size: int = 2
    candidate_k: int = 80
    top_k: int = 10
    negative_ratio: int = 2
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))

    def with_artifact_dir(self, path: str | Path) -> "ProjectConfig":
        return ProjectConfig(**{**self.__dict__, "artifact_dir": Path(path)})

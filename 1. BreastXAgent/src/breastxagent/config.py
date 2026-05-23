"""Configuration helpers for local runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from breastxagent.constants import HF_REPO_ID


@dataclass(frozen=True)
class BreastXConfig:
    """Filesystem configuration for a BreastXAgent run."""

    repo_id: str = HF_REPO_ID
    repo_dir: Path = Path("data/breast_density_classification")
    output_dir: Path = Path("outputs")

    @property
    def model_path(self) -> Path:
        return self.repo_dir / "models" / "model.pt"

    @property
    def sample_dir(self) -> Path:
        return self.repo_dir / "sample_data"

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def as_path(value: str | Path) -> Path:
    """Return a user-provided string/path as an expanded Path."""

    return Path(value).expanduser()


"""Download utilities for the Hugging Face model bundle."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from breastxagent.constants import HF_REPO_ID


def download_model_bundle(
    local_dir: str | Path,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Download the MONAI breast-density bundle to a local directory."""

    target = Path(local_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    return target


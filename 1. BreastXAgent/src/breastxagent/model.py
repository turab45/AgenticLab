"""Model architecture and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import inception_v3


class InceptionMONAIStyleAdapter(nn.Module):
    """Inception-v3 adapter matching MONAI's TorchVisionFCModel checkpoint."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        base = inception_v3(
            weights=None,
            aux_logits=True,
            transform_input=False,
            init_weights=False,
        )
        base.fc = nn.Identity()
        self.features = base
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)

        if hasattr(features, "logits"):
            features = features.logits
        elif isinstance(features, (tuple, list)):
            features = features[0]

        features = torch.flatten(features, 1)
        return self.fc(features)


def get_device(requested: str | None = None) -> torch.device:
    """Resolve a torch device from a CLI-friendly value."""

    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dict from common checkpoint formats."""

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove wrappers commonly introduced by training frameworks."""

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("network.", "")
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a checkpoint in a way that works across torch versions."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return clean_state_dict(extract_state_dict(checkpoint))


def build_model(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
    num_classes: int = 4,
    strict: bool = False,
) -> InceptionMONAIStyleAdapter:
    """Build the model and load pretrained weights."""

    resolved_device = get_device(str(device) if device else None)
    model = InceptionMONAIStyleAdapter(num_classes=num_classes).to(resolved_device)
    state_dict = load_checkpoint(checkpoint_path, resolved_device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint loading failed with strict=True. "
            f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}"
        )

    model.eval()
    return model

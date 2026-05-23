"""Prediction utilities for breast-density classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from breastxagent.constants import CLASS_DESCRIPTIONS, CLASS_NAMES, IMAGE_EXTENSIONS
from breastxagent.preprocessing import preprocess_image


@dataclass(frozen=True)
class PredictionResult:
    image: str
    true_class: str | None
    predicted_class: str
    predicted_description: str
    confidence_sigmoid: float
    sigmoid_A: float
    sigmoid_B: float
    sigmoid_C: float
    sigmoid_D: float
    softmax_A: float
    softmax_B: float
    softmax_C: float
    softmax_D: float
    correct: bool | None
    path: str


def list_images(image_dir: str | Path) -> list[Path]:
    """Return image files under a directory, including class subfolders."""

    root = Path(image_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def infer_true_class(image_path: str | Path) -> str | None:
    """Infer label from parent folder when it is one of A/B/C/D."""

    parent = Path(image_path).parent.name
    return parent if parent in CLASS_NAMES else None


class BreastDensityPredictor:
    """Reusable prediction wrapper around a loaded torch model."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict_image(self, image_path: str | Path) -> PredictionResult:
        tensor = preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            sigmoid_scores = torch.sigmoid(logits)[0].detach().cpu().numpy()
            softmax_scores = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(sigmoid_scores))
        pred_class = CLASS_NAMES[pred_idx]
        true_class = infer_true_class(image_path)

        return PredictionResult(
            image=Path(image_path).name,
            true_class=true_class,
            predicted_class=pred_class,
            predicted_description=CLASS_DESCRIPTIONS[pred_class],
            confidence_sigmoid=float(sigmoid_scores[pred_idx]),
            sigmoid_A=float(sigmoid_scores[0]),
            sigmoid_B=float(sigmoid_scores[1]),
            sigmoid_C=float(sigmoid_scores[2]),
            sigmoid_D=float(sigmoid_scores[3]),
            softmax_A=float(softmax_scores[0]),
            softmax_B=float(softmax_scores[1]),
            softmax_C=float(softmax_scores[2]),
            softmax_D=float(softmax_scores[3]),
            correct=(true_class == pred_class) if true_class else None,
            path=str(Path(image_path)),
        )

    def predict_paths(self, image_paths: list[str | Path]) -> pd.DataFrame:
        rows = [asdict(self.predict_image(path)) for path in image_paths]
        return pd.DataFrame(rows)

    def predict_directory(self, image_dir: str | Path) -> pd.DataFrame:
        return self.predict_paths(list_images(image_dir))


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Compute metrics when ground-truth labels are available."""

    labelled = df.dropna(subset=["true_class"])
    if labelled.empty:
        return {}

    y_true = labelled["true_class"]
    y_pred = labelled["predicted_class"]
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(CLASS_NAMES),
            target_names=list(CLASS_NAMES),
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=[f"True_{label}" for label in CLASS_NAMES],
            columns=[f"Pred_{label}" for label in CLASS_NAMES],
        ),
    }


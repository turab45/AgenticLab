"""Grad-CAM generation for the classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from breastxagent.constants import CLASS_NAMES
from breastxagent.preprocessing import preprocess_image


@dataclass(frozen=True)
class GradCAMResult:
    image_path: str
    true_class: str | None
    predicted_class: str
    confidence: float
    gradcam_path: str
    correct: bool | None


class GradCAM:
    """Generate Grad-CAM heatmaps for a target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_inputs, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        image_tensor.requires_grad_(True)
        logits = self.model(image_tensor)

        if target_class_idx is None:
            scores = torch.sigmoid(logits)
            target_class_idx = int(torch.argmax(scores, dim=1).item())

        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations.")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device,
        )
        for idx, weight in enumerate(weights):
            cam += weight * activations[idx]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), target_class_idx


def overlay_cam_on_image(
    image_path: str | Path,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize a CAM and overlay it on the original image."""

    original = Image.open(Path(image_path)).convert("RGB")
    original_np = np.asarray(original).astype(np.float32) / 255.0

    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_img = cam_img.resize(original.size, Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

    heatmap = plt.get_cmap("jet")(cam_resized)[:, :, :3]
    overlay = (1 - alpha) * original_np + alpha * heatmap
    return original_np, heatmap, np.clip(overlay, 0, 1)


def save_gradcam_figure(
    image_path: str | Path,
    cam: np.ndarray,
    predicted_class: str,
    confidence: float,
    output_path: str | Path,
) -> Path:
    """Save a three-panel Grad-CAM figure."""

    original_np, heatmap, overlay = overlay_cam_on_image(image_path, cam)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    panels = [
        (original_np, "Original Image"),
        (heatmap, "Grad-CAM Heatmap"),
        (overlay, f"Overlay\nPred: {predicted_class}, Conf: {confidence:.2f}"),
    ]
    for index, (image, title) in enumerate(panels, start=1):
        plt.subplot(1, 3, index)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    return output


def generate_gradcam_for_prediction(
    model: torch.nn.Module,
    device: torch.device,
    prediction: pd.Series,
    output_dir: str | Path,
) -> GradCAMResult:
    """Generate a Grad-CAM image for one prediction row."""

    target_layer = model.features.Mixed_7c
    gradcam = GradCAM(model, target_layer)

    image_path = Path(prediction["path"])
    tensor = preprocess_image(image_path).to(device)

    predicted_class = str(prediction["predicted_class"])
    pred_idx = CLASS_NAMES.index(predicted_class)
    confidence = float(prediction["confidence_sigmoid"])

    try:
        cam, _target_idx = gradcam.generate(tensor, target_class_idx=pred_idx)
    finally:
        gradcam.remove_hooks()

    base_name = image_path.stem
    gradcam_path = Path(output_dir) / f"{base_name}_pred_{predicted_class}_gradcam.png"
    save_gradcam_figure(image_path, cam, predicted_class, confidence, gradcam_path)

    true_class = prediction.get("true_class")
    if pd.isna(true_class):
        true_class = None

    return GradCAMResult(
        image_path=str(image_path),
        true_class=true_class,
        predicted_class=predicted_class,
        confidence=confidence,
        gradcam_path=str(gradcam_path),
        correct=(true_class == predicted_class) if true_class else None,
    )


def generate_gradcam_dataframe(
    model: torch.nn.Module,
    device: torch.device,
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    rows = [
        asdict(generate_gradcam_for_prediction(model, device, row, output_dir))
        for _idx, row in predictions_df.iterrows()
    ]
    return pd.DataFrame(rows)


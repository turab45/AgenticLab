"""Image preprocessing used by the model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def preprocess_image(image_path: str | Path, image_size: int = 299) -> torch.Tensor:
    """Load and normalize an image into a BCHW float tensor."""

    image = Image.open(Path(image_path)).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)

    array = np.asarray(image).astype(np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    array = (array - array_min) / (array_max - array_min + 1e-8)

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()


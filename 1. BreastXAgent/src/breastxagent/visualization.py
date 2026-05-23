"""Plotting helpers for notebooks and exploratory runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def show_predictions(df: pd.DataFrame, n: int = 8) -> None:
    sample_df = df.head(min(n, len(df)))
    cols = 4
    rows = int(np.ceil(len(sample_df) / cols))

    plt.figure(figsize=(16, 4 * rows))
    for index, row in enumerate(sample_df.itertuples(), start=1):
        image = Image.open(Path(row.path)).convert("RGB")
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(
            f"True: {row.true_class}\n"
            f"Pred: {row.predicted_class} | Conf: {row.confidence_sigmoid:.2f}"
        )
    plt.tight_layout()
    plt.show()


def show_gradcam_examples(gradcam_df: pd.DataFrame, n: int = 8) -> None:
    sample_df = gradcam_df.head(min(n, len(gradcam_df)))
    cols = 2
    rows = int(np.ceil(len(sample_df) / cols))

    plt.figure(figsize=(14, 5 * rows))
    for index, row in enumerate(sample_df.itertuples(), start=1):
        image = Image.open(Path(row.gradcam_path)).convert("RGB")
        plt.subplot(rows, cols, index)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {row.true_class} | "
            f"Pred: {row.predicted_class} | "
            f"Conf: {row.confidence:.2f}"
        )
    plt.tight_layout()
    plt.show()


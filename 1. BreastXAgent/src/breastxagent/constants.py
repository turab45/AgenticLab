"""Shared constants for breast-density classification."""

HF_REPO_ID = "MONAI/breast_density_classification"

CLASS_NAMES = ("A", "B", "C", "D")

CLASS_DESCRIPTIONS = {
    "A": "Almost entirely fatty",
    "B": "Scattered fibroglandular density",
    "C": "Heterogeneously dense",
    "D": "Extremely dense",
}

DENSITY_DESCRIPTIONS = {
    "A": "almost entirely fatty breast tissue",
    "B": "scattered fibroglandular breast density",
    "C": "heterogeneously dense breast tissue",
    "D": "extremely dense breast tissue",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


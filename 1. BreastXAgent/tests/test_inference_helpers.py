from pathlib import Path

from breastxagent.inference import infer_true_class


def test_infer_true_class_from_parent_folder():
    assert infer_true_class(Path("sample_data/A/image.png")) == "A"
    assert infer_true_class(Path("sample_data/unknown/image.png")) is None


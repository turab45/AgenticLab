from breastxagent.agent import (
    confidence_level,
    generate_report,
    report_has_forbidden_claims,
)


class UnsafeClient:
    def generate(self, prompt: str) -> str:
        return "The heatmap shows regions of interest and recommends diagnosis."


class SafeClient:
    def generate(self, prompt: str) -> str:
        return "1. Prediction Summary\nThe provided facts were formatted only."


def test_confidence_level_thresholds():
    assert confidence_level(0.86) == "high"
    assert confidence_level(0.70) == "moderate"
    assert confidence_level(0.69) == "low"


def test_forbidden_claim_detection():
    assert report_has_forbidden_claims("The patient needs further investigation.")
    assert not report_has_forbidden_claims("The confidence level is moderate.")


def test_unsafe_llm_output_falls_back():
    report = generate_report(
        image_path="sample.png",
        gradcam_path="sample_gradcam.png",
        predicted_class="B",
        confidence=0.9,
        llm_client=UnsafeClient(),
    )
    assert report.used_fallback is True
    assert "sample_gradcam.png" in report.final_report


def test_safe_llm_output_is_used():
    report = generate_report(
        image_path="sample.png",
        gradcam_path="sample_gradcam.png",
        predicted_class="B",
        confidence=0.9,
        llm_client=SafeClient(),
    )
    assert report.used_fallback is False
    assert report.final_report.startswith("1. Prediction Summary")


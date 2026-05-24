"""Guarded report-generation layer."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, TypedDict

from breastxagent.constants import DENSITY_DESCRIPTIONS


class BreastDensityAgentState(TypedDict, total=False):
    image_path: str
    gradcam_path: str
    predicted_class: str
    confidence: float
    density_description: Optional[str]
    confidence_level: Optional[str]
    evidence_status: Optional[str]
    recommendation_level: Optional[str]
    llm_prompt: Optional[str]
    final_report: Optional[str]
    used_fallback: Optional[bool]


class ChatReportClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate a report from a prompt."""


@dataclass(frozen=True)
class AgentReport:
    image_path: str
    gradcam_path: str
    predicted_class: str
    confidence: float
    density_description: str
    confidence_level: str
    evidence_status: str
    recommendation_level: str
    used_fallback: bool
    final_report: str


FORBIDDEN_TERMS = (
    "the image contains",
    "heatmap shows",
    "heatmap highlights tissue",
    "regions of interest",
    "additional imaging",
    "further investigation",
    "patient",
    "diagnosis",
    "cancer",
    "malignant",
    "benign",
    "lesion",
    "abnormality",
    "treatment",
    "clinical action",
    "clinical recommendation",
    "medical recommendation",
    "disease",
    "tumor",
    "carcinoma",
)


def confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "low"


def evidence_routing(level: str) -> tuple[str, str]:
    if level == "high":
        return (
            "Grad-CAM visual evidence was generated for the predicted class. "
            "The classifier confidence is high, but the Grad-CAM map is only an "
            "explanatory visualization, not proof of the prediction.",
            "standard evidence-guided research report",
        )
    if level == "moderate":
        return (
            "Grad-CAM visual evidence was generated for the predicted class. "
            "The classifier confidence is moderate, so the result should be "
            "interpreted cautiously.",
            "cautious evidence-guided research report",
        )
    return (
        "Grad-CAM visual evidence was generated, but the classifier confidence is low. "
        "The prediction should be treated as uncertain.",
        "low-confidence research warning",
    )


def report_has_forbidden_claims(report: str) -> bool:
    report_lower = report.lower()
    return any(term in report_lower for term in FORBIDDEN_TERMS)


def make_llm_prompt(state: BreastDensityAgentState) -> str:
    image_name = os.path.basename(state["image_path"])
    gradcam_name = os.path.basename(state["gradcam_path"])

    return f"""
You are a research-writing assistant. Your only task is to format the provided model outputs into a short report.

You are NOT allowed to interpret the medical image.
You are NOT allowed to interpret the Grad-CAM heatmap content.
You are NOT allowed to describe visible tissue, lesion, abnormality, cancer, disease, or patient outcome.
You are NOT allowed to recommend further imaging, investigation, diagnosis, treatment, or clinical action.

Allowed facts:
- Image name: {image_name}
- The model predicted breast density class: {state["predicted_class"]}
- The density class corresponds to: {state["density_description"]}
- The model confidence score is: {state["confidence"]:.3f}
- The confidence level is: {state["confidence_level"]}
- A Grad-CAM heatmap file was generated: {gradcam_name}
- Evidence status: {state["evidence_status"]}
- Agent decision: {state["recommendation_level"]}

Write exactly four short sections:

1. Prediction Summary
2. Visual Evidence
3. Confidence Assessment
4. Research Note

Rules:
- Keep each section 1 to 2 sentences only.
- Do not add information beyond the allowed facts.
- Do not claim that Grad-CAM proves the prediction.
- The output must be suitable for a research paper, not a clinical report.
""".strip()


def make_safe_fallback_report(state: BreastDensityAgentState) -> str:
    image_name = os.path.basename(state["image_path"])
    gradcam_name = os.path.basename(state["gradcam_path"])

    return f"""
1. Prediction Summary
The model classified {image_name} as breast density class {state["predicted_class"]}, corresponding to {state["density_description"]}. The confidence score was {state["confidence"]:.3f}.

2. Visual Evidence
A Grad-CAM heatmap was generated for the predicted class and saved as {gradcam_name}. The heatmap is used only as a visual explanation of the model decision process, not as proof of the prediction.

3. Confidence Assessment
The confidence level is {state["confidence_level"]}. Based on the agent decision, this case is reported as: {state["recommendation_level"]}.

4. Research Note
This output is intended for research and explainability support only. It is not a clinical diagnosis or medical recommendation.
""".strip()


def build_state(
    image_path: str,
    gradcam_path: str,
    predicted_class: str,
    confidence: float,
) -> BreastDensityAgentState:
    level = confidence_level(float(confidence))
    evidence_status, recommendation_level = evidence_routing(level)
    density_description = DENSITY_DESCRIPTIONS.get(
        predicted_class,
        "unknown breast density category",
    )

    state: BreastDensityAgentState = {
        "image_path": image_path,
        "gradcam_path": gradcam_path,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "density_description": density_description,
        "confidence_level": level,
        "evidence_status": evidence_status,
        "recommendation_level": recommendation_level,
        "llm_prompt": None,
        "final_report": None,
        "used_fallback": False,
    }
    state["llm_prompt"] = make_llm_prompt(state)
    return state


def interpret_prediction_node(state: BreastDensityAgentState) -> dict[str, str]:
    return {
        "density_description": DENSITY_DESCRIPTIONS.get(
            state["predicted_class"],
            "unknown breast density category",
        )
    }


def confidence_analysis_node(state: BreastDensityAgentState) -> dict[str, str]:
    return {"confidence_level": confidence_level(float(state["confidence"]))}


def confidence_router(state: BreastDensityAgentState) -> str:
    return "uncertain" if state["confidence_level"] == "low" else "supported"


def evidence_supported_node(state: BreastDensityAgentState) -> dict[str, str]:
    evidence_status, recommendation_level = evidence_routing(
        state["confidence_level"] or "low"
    )
    return {
        "evidence_status": evidence_status,
        "recommendation_level": recommendation_level,
    }


def uncertain_node(state: BreastDensityAgentState) -> dict[str, str]:
    evidence_status, recommendation_level = evidence_routing("low")
    return {
        "evidence_status": evidence_status,
        "recommendation_level": recommendation_level,
    }


def build_llm_prompt_node(state: BreastDensityAgentState) -> dict[str, str]:
    return {"llm_prompt": make_llm_prompt(state)}


def generate_report(
    image_path: str,
    gradcam_path: str,
    predicted_class: str,
    confidence: float,
    llm_client: ChatReportClient | None = None,
) -> AgentReport:
    state = build_state(image_path, gradcam_path, predicted_class, confidence)
    fallback_report = make_safe_fallback_report(state)

    used_fallback = llm_client is None
    final_report = fallback_report

    if llm_client is not None:
        candidate = llm_client.generate(state["llm_prompt"] or "")
        if candidate and not report_has_forbidden_claims(candidate):
            final_report = candidate
            used_fallback = False
        else:
            used_fallback = True

    return AgentReport(
        image_path=image_path,
        gradcam_path=gradcam_path,
        predicted_class=predicted_class,
        confidence=float(confidence),
        density_description=state["density_description"] or "",
        confidence_level=state["confidence_level"] or "",
        evidence_status=state["evidence_status"] or "",
        recommendation_level=state["recommendation_level"] or "",
        used_fallback=used_fallback,
        final_report=final_report,
    )


def build_langgraph_agent(llm_client: ChatReportClient | None = None):
    """Build the optional LangGraph workflow from the original notebook design."""

    from langgraph.graph import END, START, StateGraph

    def report_node(state: BreastDensityAgentState) -> dict[str, object]:
        fallback_report = make_safe_fallback_report(state)

        if llm_client is None:
            return {"final_report": fallback_report, "used_fallback": True}

        candidate = llm_client.generate(state["llm_prompt"] or "")
        if candidate and not report_has_forbidden_claims(candidate):
            return {"final_report": candidate, "used_fallback": False}
        return {"final_report": fallback_report, "used_fallback": True}

    workflow = StateGraph(BreastDensityAgentState)
    workflow.add_node("interpret_prediction", interpret_prediction_node)
    workflow.add_node("confidence_analysis", confidence_analysis_node)
    workflow.add_node("evidence_supported", evidence_supported_node)
    workflow.add_node("uncertain", uncertain_node)
    workflow.add_node("build_llm_prompt", build_llm_prompt_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "interpret_prediction")
    workflow.add_edge("interpret_prediction", "confidence_analysis")
    workflow.add_conditional_edges(
        "confidence_analysis",
        confidence_router,
        {"supported": "evidence_supported", "uncertain": "uncertain"},
    )
    workflow.add_edge("evidence_supported", "build_llm_prompt")
    workflow.add_edge("uncertain", "build_llm_prompt")
    workflow.add_edge("build_llm_prompt", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


class HuggingFaceChatClient:
    """Small adapter around Hugging Face chat completion."""

    def __init__(
        self,
        model: str,
        token: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(model=model, token=token or os.getenv("HF_TOKEN"))
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict research-report formatting assistant. "
                    "Do not add clinical or visual interpretations beyond the provided facts."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for _attempt in range(self.max_retries):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=220,
                    temperature=0.0,
                    top_p=0.8,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                time.sleep(2)

        return ""


def generate_reports_dataframe(
    gradcam_df,
    llm_client: ChatReportClient | None = None,
):
    import pandas as pd

    rows = []
    for row in gradcam_df.itertuples(index=False):
        report = generate_report(
            image_path=row.image_path,
            gradcam_path=row.gradcam_path,
            predicted_class=row.predicted_class,
            confidence=float(row.confidence),
            llm_client=llm_client,
        )
        rows.append(asdict(report))
    return pd.DataFrame(rows)


def load_gradcam_csv(path: str | Path) -> pd.DataFrame:
    import pandas as pd

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Grad-CAM CSV not found: {csv_path}")
    return pd.read_csv(csv_path)

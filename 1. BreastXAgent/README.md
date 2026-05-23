# BreastXAgent

BreastXAgent is a research-oriented pipeline for breast density classification using the MONAI `breast_density_classification` model bundle, Grad-CAM visual explanation, and a guarded report-generation layer.

The original notebook has been preserved at `notebooks/breastxagent_original.ipynb`. The reusable code now lives in `src/breastxagent`.

> Research use only. This project is not intended for clinical diagnosis, medical recommendation, or patient-facing decision making.

## Project Structure

```text
.
├── notebooks/
│   └── breastxagent_original.ipynb
├── src/
│   └── breastxagent/
│       ├── agent.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── gradcam.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocessing.py
│       └── visualization.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you only want the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

## Download the Model Bundle

```bash
breastxagent download --local-dir data/breast_density_classification
```

This downloads `MONAI/breast_density_classification` from Hugging Face.

## Run Inference

Use the sample images included in the downloaded model bundle:

```bash
breastxagent infer \
  --repo-dir data/breast_density_classification \
  --output-csv outputs/predictions.csv
```

Or point to your own image folder:

```bash
breastxagent infer \
  --repo-dir data/breast_density_classification \
  --image-dir path/to/images \
  --output-csv outputs/predictions.csv
```

If your images are organized as `A/`, `B/`, `C/`, and `D/` subfolders, the CLI will treat each parent folder name as the ground-truth class and include metrics.

## Generate Grad-CAM Outputs

```bash
breastxagent gradcam \
  --repo-dir data/breast_density_classification \
  --predictions-csv outputs/predictions.csv \
  --output-dir outputs/gradcam \
  --output-csv outputs/gradcam_results.csv
```

## Generate Research Reports

Deterministic guarded reports:

```bash
breastxagent report \
  --gradcam-csv outputs/gradcam_results.csv \
  --output-csv outputs/agent_reports.csv
```

Optional Hugging Face LLM formatting:

```bash
export HF_TOKEN=your_token
breastxagent report \
  --gradcam-csv outputs/gradcam_results.csv \
  --output-csv outputs/llm_agent_reports.csv \
  --use-llm \
  --hf-model meta-llama/Llama-3.1-8B-Instruct
```

The LLM output is checked for forbidden clinical or visual interpretation claims. If a report violates the guardrails, the pipeline falls back to the deterministic report.

## Development Checks

```bash
python -m compileall src tests
pytest
```

## Notes

- The package recreates the MONAI TorchVision Inception-v3 classifier architecture without requiring MONAI.
- Inference uses sigmoid scores for the primary prediction to match the original notebook behavior.
- Grad-CAM uses the Inception-v3 `Mixed_7c` block by default.
- Generated data, downloaded weights, and outputs are ignored by Git.


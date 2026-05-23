# BreastXAgent

BreastXAgent is a research-oriented pipeline for breast density classification using the MONAI `breast_density_classification` model bundle, Grad-CAM visual explanation, and a guarded report-generation layer.

The original notebook has been preserved at `notebooks/breastxagent_original.ipynb`. The reusable code now lives in `src/breastxagent`.

> Research use only. This project is not intended for clinical diagnosis, medical recommendation, or patient-facing decision making.

## Project Structure

```text
.
├── notebooks/
│   └── breastxagent_original.ipynb
├── scripts/
│   ├── download_model.sh
│   ├── run_gradcam.sh
│   ├── run_inference.sh
│   ├── run_llm_reports.sh
│   ├── run_reports.sh
│   └── setup.sh
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

Recommended setup from a fresh clone:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

This installs the runtime dependencies from `requirements.txt`. You do not need the `breastxagent` shell command to be globally installed.

For development tools such as `pytest` and `ruff`, run:

```bash
python3 -m pip install -e ".[dev]"
```

## Download the Model Bundle

```bash
bash scripts/download_model.sh
```

This downloads `MONAI/breast_density_classification` from Hugging Face.

To download to a different directory:

```bash
bash scripts/download_model.sh path/to/model_bundle
```

## Run Inference

Use the sample images included in the downloaded model bundle:

```bash
bash scripts/run_inference.sh
```

Or point to your own image folder:

```bash
IMAGE_DIR=path/to/images bash scripts/run_inference.sh
```

If your images are organized as `A/`, `B/`, `C/`, and `D/` subfolders, the pipeline will treat each parent folder name as the ground-truth class and include metrics.

Optional overrides:

```bash
REPO_DIR=data/breast_density_classification \
OUTPUT_CSV=outputs/predictions.csv \
DEVICE=cpu \
bash scripts/run_inference.sh
```

## Generate Grad-CAM Outputs

```bash
bash scripts/run_gradcam.sh
```

## Generate Research Reports

Deterministic guarded reports:

```bash
bash scripts/run_reports.sh
```

Optional Hugging Face LLM formatting:

```bash
export HF_TOKEN=your_token
bash scripts/run_llm_reports.sh
```

The LLM output is checked for forbidden clinical or visual interpretation claims. If a report violates the guardrails, the pipeline falls back to the deterministic report.

## Optional CLI

The project still includes a Python CLI because the bash scripts use it internally through:

```bash
PYTHONPATH=src python3 -m breastxagent.cli ...
```

You only need the shorter `breastxagent ...` command if you install the package in editable mode:

```bash
python3 -m pip install -e .
breastxagent --help
```

If that command is not recognized on a new machine, use the bash scripts instead.

## Development Checks

```bash
python3 -m compileall src tests
python3 -m pytest
```

## Notes

- The package recreates the MONAI TorchVision Inception-v3 classifier architecture without requiring MONAI.
- Inference uses sigmoid scores for the primary prediction to match the original notebook behavior.
- Grad-CAM uses the Inception-v3 `Mixed_7c` block by default.
- Generated data, downloaded weights, and outputs are ignored by Git.

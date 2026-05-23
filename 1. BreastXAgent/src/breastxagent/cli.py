"""Command-line interface for BreastXAgent."""

from __future__ import annotations

import argparse

from breastxagent.config import BreastXConfig, as_path


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-dir",
        default=str(BreastXConfig().repo_dir),
        help="Downloaded MONAI bundle directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc.",
    )


def download_command(args: argparse.Namespace) -> None:
    from breastxagent.download import download_model_bundle

    target = download_model_bundle(args.local_dir, repo_id=args.repo_id)
    print(f"Downloaded model bundle to: {target}")


def infer_command(args: argparse.Namespace) -> None:
    from breastxagent.inference import BreastDensityPredictor, compute_metrics
    from breastxagent.model import build_model, get_device

    repo_dir = as_path(args.repo_dir)
    config = BreastXConfig(repo_dir=repo_dir, output_dir=as_path(args.output_dir))

    image_dir = as_path(args.image_dir) if args.image_dir else config.sample_dir
    device = get_device(args.device)
    model = build_model(config.model_path, device=device)
    predictor = BreastDensityPredictor(model, device)

    df = predictor.predict_directory(image_dir)
    output_csv = as_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Predictions saved to: {output_csv}")
    print(f"Images processed: {len(df)}")

    metrics = compute_metrics(df)
    if metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification report:")
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(metrics["confusion_matrix"].to_string())


def gradcam_command(args: argparse.Namespace) -> None:
    import pandas as pd

    from breastxagent.gradcam import generate_gradcam_dataframe
    from breastxagent.model import build_model, get_device

    repo_dir = as_path(args.repo_dir)
    config = BreastXConfig(repo_dir=repo_dir)
    device = get_device(args.device)
    model = build_model(config.model_path, device=device)

    predictions_df = pd.read_csv(as_path(args.predictions_csv))
    gradcam_df = generate_gradcam_dataframe(
        model=model,
        device=device,
        predictions_df=predictions_df,
        output_dir=as_path(args.output_dir),
    )

    output_csv = as_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gradcam_df.to_csv(output_csv, index=False)

    print(f"Grad-CAM CSV saved to: {output_csv}")
    print(f"Grad-CAM images saved in: {as_path(args.output_dir)}")


def report_command(args: argparse.Namespace) -> None:
    import pandas as pd

    from breastxagent.agent import HuggingFaceChatClient, generate_reports_dataframe

    gradcam_df = pd.read_csv(as_path(args.gradcam_csv))
    llm_client = None

    if args.use_llm:
        llm_client = HuggingFaceChatClient(model=args.hf_model, token=args.hf_token)

    reports_df = generate_reports_dataframe(gradcam_df, llm_client=llm_client)
    output_csv = as_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reports_df.to_csv(output_csv, index=False)

    fallback_count = int(reports_df["used_fallback"].sum())
    print(f"Reports saved to: {output_csv}")
    print(f"Fallback reports used: {fallback_count}/{len(reports_df)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breastxagent",
        description="Breast density classification, Grad-CAM, and guarded reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download model bundle.")
    download_parser.add_argument("--repo-id", default=BreastXConfig().repo_id)
    download_parser.add_argument(
        "--local-dir",
        default=str(BreastXConfig().repo_dir),
        help="Where to download the model bundle.",
    )
    download_parser.set_defaults(func=download_command)

    infer_parser = subparsers.add_parser("infer", help="Run classifier inference.")
    add_common_model_args(infer_parser)
    infer_parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory of images. Defaults to repo sample_data.",
    )
    infer_parser.add_argument("--output-dir", default="outputs")
    infer_parser.add_argument("--output-csv", default="outputs/predictions.csv")
    infer_parser.set_defaults(func=infer_command)

    gradcam_parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM files.")
    add_common_model_args(gradcam_parser)
    gradcam_parser.add_argument(
        "--predictions-csv",
        default="outputs/predictions.csv",
        help="CSV produced by the infer command.",
    )
    gradcam_parser.add_argument("--output-dir", default="outputs/gradcam")
    gradcam_parser.add_argument("--output-csv", default="outputs/gradcam_results.csv")
    gradcam_parser.set_defaults(func=gradcam_command)

    report_parser = subparsers.add_parser("report", help="Generate guarded reports.")
    report_parser.add_argument("--gradcam-csv", default="outputs/gradcam_results.csv")
    report_parser.add_argument("--output-csv", default="outputs/agent_reports.csv")
    report_parser.add_argument("--use-llm", action="store_true")
    report_parser.add_argument(
        "--hf-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face chat model used when --use-llm is set.",
    )
    report_parser.add_argument("--hf-token", default=None)
    report_parser.set_defaults(func=report_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

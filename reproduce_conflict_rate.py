"""One-command driver for the FEVER spurious-conflict experiment.

Two modes are supported:

1. Supply two existing ``predictions.jsonl`` files and run only the audited
   analysis.
2. Supply the Det2Ver and CVPR-FV adapter checkpoints. The driver runs an
   evaluation-only FEVER pass for each model and then performs the analysis.

The model checkpoints are deliberately required inputs; this script never
substitutes reported numbers for actual model outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve().parent
DET2VER_DIR = HERE / "Det2Ver"


def run(command: List[str], cwd: Path) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n[{cwd.name}] {printable}\n", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def evaluate_from_checkpoints(args: argparse.Namespace) -> tuple[Path, Path]:
    det_model_path = DET2VER_DIR / "model.py"
    det_reader_path = DET2VER_DIR / "data_reader.py"
    if not det_model_path.is_file() or not det_reader_path.is_file():
        raise FileNotFoundError(
            f"expected the vendored Det2Ver implementation at {DET2VER_DIR}"
        )
    det_model_source = det_model_path.read_text(encoding="utf-8")
    det_reader_source = det_reader_path.read_text(encoding="utf-8")
    if "predictions.jsonl" not in det_model_source or "'instance_id'" not in det_reader_source:
        raise RuntimeError(
            "the vendored Det2Ver implementation lacks per-instance export support"
        )

    det_checkpoint = args.det2ver_checkpoint.resolve()
    cvpr_checkpoint = args.cvpr_checkpoint.resolve()
    for name, path in (
        ("Det2Ver checkpoint", det_checkpoint),
        ("CVPR-FV checkpoint", cvpr_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    run_root = args.run_root.resolve()
    det_root = run_root / "Det2Ver"
    cvpr_root = run_root / "CVPR_FV"
    exp_name = "fever_K4_seed0_conflict"

    # Training-only datasets are disabled to make evaluation cheaper. This
    # does not alter the fixed adapter weights or the FEVER validation loader.
    run([
        args.python,
        "train.py",
        "--dataset", "fever",
        "--shot_num", "4",
        "--seed", "0",
        "--few_shot", "true",
        "--zero_shot", "true",
        "--use_rumor_detection", "false",
        "--rd_total_per_dataset", "0",
        "--eval_only", "true",
        "--save_model", "false",
        "--load_weight", str(det_checkpoint),
        "--exp_root", str(det_root),
        "--exp_name", exp_name,
        "--eval_batch_size", str(args.eval_batch_size),
        "--precision", args.precision,
        "--accelerator", args.accelerator,
        "--devices", args.devices,
    ], DET2VER_DIR)

    run([
        args.python,
        "train.py",
        "--dataset", "fever",
        "--shot_num", "4",
        "--seed", "0",
        "--few_shot", "true",
        "--zero_shot", "true",
        "--backbone", "t0-3b",
        "--use_cvp", "true",
        "--cvp_total_per_dataset", "0",
        "--lam_prior", "0.5",
        "--nei_floor_gamma", "0.1",
        "--eval_only", "true",
        "--save_model", "false",
        "--load_weight", str(cvpr_checkpoint),
        "--exp_root", str(cvpr_root),
        "--exp_name", exp_name,
        "--eval_batch_size", str(args.eval_batch_size),
        "--precision", args.precision,
        "--accelerator", args.accelerator,
        "--devices", args.devices,
    ], HERE)

    return (
        det_root / exp_name / "predictions.jsonl",
        cvpr_root / exp_name / "predictions.jsonl",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and reproduce the FEVER spurious-conflict table."
    )
    inputs = parser.add_argument_group("input mode")
    inputs.add_argument("--det2ver-pred", type=Path)
    inputs.add_argument("--cvpr-pred", type=Path)
    inputs.add_argument("--det2ver-checkpoint", type=Path)
    inputs.add_argument("--cvpr-checkpoint", type=Path)

    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--run-root", type=Path, default=HERE / "output" / "conflict_reproduction"
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=["16", "32", "bf16"], default="32")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--output", type=Path, default=HERE / "RESULTS" / "conflict_rate_reproduced.md"
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=HERE / "RESULTS" / "conflict_rate_instances.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=HERE / "RESULTS" / "conflict_rate_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_mode = args.det2ver_pred is not None or args.cvpr_pred is not None
    checkpoint_mode = (
        args.det2ver_checkpoint is not None or args.cvpr_checkpoint is not None
    )
    if prediction_mode and checkpoint_mode:
        raise ValueError("choose prediction-file mode or checkpoint mode, not both")
    if prediction_mode:
        if args.det2ver_pred is None or args.cvpr_pred is None:
            raise ValueError("prediction mode requires both --det2ver-pred and --cvpr-pred")
        det_prediction = args.det2ver_pred.resolve()
        cvpr_prediction = args.cvpr_pred.resolve()
    elif checkpoint_mode:
        if args.det2ver_checkpoint is None or args.cvpr_checkpoint is None:
            raise ValueError(
                "checkpoint mode requires both --det2ver-checkpoint and --cvpr-checkpoint"
            )
        det_prediction, cvpr_prediction = evaluate_from_checkpoints(args)
    else:
        raise ValueError(
            "provide either both prediction files or both trained adapter checkpoints"
        )

    for name, path in (
        ("Det2Ver predictions", det_prediction),
        ("CVPR-FV predictions", cvpr_prediction),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} do not exist: {path}")

    run([
        args.python,
        str(HERE / "analyze_conflict_rate.py"),
        "--det2ver-pred", str(det_prediction),
        "--cvpr-pred", str(cvpr_prediction),
        "--expected-instances", "9985",
        "--output", str(args.output.resolve()),
        "--details-output", str(args.details_output.resolve()),
        "--summary-json", str(args.summary_json.resolve()),
    ], HERE)


if __name__ == "__main__":
    main()

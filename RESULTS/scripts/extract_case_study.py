#!/usr/bin/env python3
"""Extract the five reported qualitative cases from model prediction exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CASES = ("fever:58686", "fever:87782", "fever:114625", "fever:185758", "scifact:756")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def index_by_id(path: Path) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = str(row.get("instance_id", row.get("id", "")))
        if not key:
            raise ValueError(f"row without instance_id/id in {path}")
        if key in indexed:
            raise ValueError(f"duplicate instance ID {key!r} in {path}")
        indexed[key] = row
    return indexed


def parse_mapping(values: list[str], option: str) -> dict[str, Path]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects DATASET=PATH, got {value!r}")
        dataset, raw_path = value.split("=", 1)
        mapping[dataset.strip().lower()] = Path(raw_path)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det2ver-pred", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--cvpr-pred", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--case", action="append", dest="cases", metavar="DATASET:ID")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    det_paths = parse_mapping(args.det2ver_pred, "--det2ver-pred")
    cvpr_paths = parse_mapping(args.cvpr_pred, "--cvpr-pred")
    cases = args.cases or list(DEFAULT_CASES)

    indexes: dict[str, tuple[dict[str, dict], dict[str, dict], dict[str, dict]]] = {}
    output_rows = []
    for case in cases:
        if ":" not in case:
            raise ValueError(f"--case expects DATASET:ID, got {case!r}")
        dataset, instance_id = case.split(":", 1)
        dataset = dataset.lower()
        if dataset not in det_paths or dataset not in cvpr_paths:
            raise ValueError(f"prediction paths were not supplied for dataset {dataset!r}")
        if dataset not in indexes:
            data_path = args.data_root / f"{dataset}_validation.jsonl"
            indexes[dataset] = (
                index_by_id(det_paths[dataset]),
                index_by_id(cvpr_paths[dataset]),
                index_by_id(data_path),
            )
        det, cvpr, data = indexes[dataset]
        missing = [name for name, index in (("Det2Ver", det), ("CVPR-FV", cvpr), ("data", data)) if instance_id not in index]
        if missing:
            raise KeyError(f"{dataset}:{instance_id} is missing from {', '.join(missing)}")

        det_row, cvpr_row, data_row = det[instance_id], cvpr[instance_id], data[instance_id]
        if str(det_row.get("gold_label")) != str(cvpr_row.get("gold_label")):
            raise ValueError(f"gold-label mismatch for {dataset}:{instance_id}")
        output_rows.append({
            "dataset": dataset,
            "instance_id": instance_id,
            "claim": data_row.get("claim", cvpr_row.get("claim", "")),
            "evidence": data_row.get("gold_evidence_text", ""),
            "gold_label": cvpr_row.get("gold_label"),
            "v": cvpr_row.get("v"),
            "q_true": cvpr_row.get("q_true"),
            "q_false": cvpr_row.get("q_false"),
            "q_uncertain": cvpr_row.get("q_uncertain"),
            "det2ver_prediction": det_row.get("prediction"),
            "cvpr_fv_prediction": cvpr_row.get("prediction"),
            "det2ver_conflict": det_row.get("lookup_conflict"),
            "cvpr_fv_conflict": cvpr_row.get("lookup_conflict"),
            "decision_scores_support_refute_nei": cvpr_row.get("decision_scores_support_refute_nei"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(output_rows)} cases to {args.output}")


if __name__ == "__main__":
    main()

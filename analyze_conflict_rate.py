"""Reproduce the FEVER verifiability-stratified conflict analysis.

The script consumes the per-instance ``predictions.jsonl`` files exported by
Det2Ver and CVPR-FV. It joins on the source dataset ID when available,
recomputes the lookup-table mismatch from the three Yes-probabilities, checks
all redundant exported fields, and reports conflict counts/rates and NEI-F1.

Example
-------
python analyze_conflict_rate.py \
  --det2ver-pred ../Det2Ver/output/fever_K4_seed0/predictions.jsonl \
  --cvpr-pred output/fever_K4_seed0/predictions.jsonl \
  --expected-instances 9985 \
  --output RESULTS/conflict_rate_reproduced.md \
  --details-output RESULTS/conflict_rate_instances.jsonl \
  --summary-json RESULTS/conflict_rate_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


STRATA: List[Tuple[str, float, float]] = [
    ("v in [0, 0.3)", 0.0, 0.3),
    ("v in [0.3, 0.6)", 0.3, 0.6),
    ("v in [0.6, 1.0]", 0.6, 1.0 + 1e-12),
]

# Answer indices follow both codebases: 0 = Yes and 1 = No. The three
# positions are true, uncertain, and false, respectively. A valid Det2Ver
# lookup row contains exactly one Yes answer.
VALID_LOOKUP_ROWS = {(0, 1, 1), (1, 0, 1), (1, 1, 0)}
VALID_LABELS = {"SUPPORT", "REFUTE", "NEI"}


def load_jsonl(path: Path) -> List[dict]:
    """Load a non-empty JSONL prediction file with useful error locations."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object in {path}:{line_number}")
            row["_source_line"] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def choose_join_field(det_rows: Sequence[dict], cvpr_rows: Sequence[dict]) -> str:
    """Prefer the source dataset ID; retain compatibility with older exports."""
    if all("instance_id" in row for row in (*det_rows, *cvpr_rows)):
        return "instance_id"
    if all("instance_idx" in row for row in (*det_rows, *cvpr_rows)):
        print(
            "warning: source instance_id is absent; falling back to row-order "
            "instance_idx. Regenerate predictions with the current model.py for "
            "a reorder-safe join.",
            file=sys.stderr,
        )
        return "instance_idx"
    raise ValueError(
        "prediction exports do not share a complete instance_id or instance_idx field"
    )


def canonical_id(value: object) -> str:
    """Create a type-aware, hashable representation of a JSON identifier."""
    return f"{type(value).__name__}:{json.dumps(value, sort_keys=True, ensure_ascii=False)}"


def index_rows(rows: Sequence[dict], join_field: str, path: Path) -> Dict[str, dict]:
    indexed: Dict[str, dict] = {}
    for row in rows:
        if join_field not in row:
            raise ValueError(
                f"missing {join_field!r} in {path}:{row.get('_source_line', '?')}"
            )
        key = canonical_id(row[join_field])
        if key in indexed:
            raise ValueError(
                f"duplicate {join_field}={row[join_field]!r} in "
                f"{path}:{row.get('_source_line', '?')}"
            )
        indexed[key] = row
    return indexed


def _probability(row: Mapping[str, object], key: str, source: str) -> float:
    if key not in row:
        raise ValueError(f"{source} row is missing {key!r}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} {key} is not numeric: {row[key]!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{source} {key} must be in [0, 1], got {value!r}")
    return value


def recompute_conflict(
    row: Mapping[str, object], source: str, threshold: float = 0.5
) -> Tuple[bool, List[int]]:
    """Compute the shared lookup-mismatch predicate from model probabilities.

    The exported ``lookup_conflict`` and binary-answer vector are treated as
    redundant audit fields. If present, they must agree with the recomputed
    value; the analysis never trusts them as its primary evidence.
    """
    q_true = _probability(row, "q_true", source)
    q_uncertain = _probability(row, "q_uncertain", source)
    q_false = _probability(row, "q_false", source)
    answers = [
        0 if q_true >= threshold else 1,
        0 if q_uncertain >= threshold else 1,
        0 if q_false >= threshold else 1,
    ]
    conflict = tuple(answers) not in VALID_LOOKUP_ROWS

    declared_answers = row.get("binary_answers_true_uncertain_false")
    if declared_answers is not None and list(declared_answers) != answers:
        raise ValueError(
            f"{source} exported binary answers {declared_answers!r} disagree "
            f"with probabilities at threshold {threshold}: {answers!r}"
        )
    if "lookup_conflict" in row and bool(row["lookup_conflict"]) != conflict:
        raise ValueError(
            f"{source} exported lookup_conflict={row['lookup_conflict']!r} "
            f"disagrees with the recomputed value {conflict}"
        )
    return conflict, answers


def class_f1(rows: Iterable[dict], prediction_key: str, positive: str = "NEI") -> float:
    tp = fp = fn = 0
    for row in rows:
        gold = row["gold_label"]
        pred = row[prediction_key]
        tp += gold == positive and pred == positive
        fp += gold != positive and pred == positive
        fn += gold == positive and pred != positive
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def stratum_for(v: float) -> str:
    for label, lower, upper in STRATA:
        if lower <= v < upper:
            return label
    raise ValueError(f"CVP score v must be in [0, 1], got {v!r}")


def join_predictions(
    det_rows: Mapping[str, dict],
    cvpr_rows: Mapping[str, dict],
    join_field: str,
    threshold: float = 0.5,
) -> List[dict]:
    det_ids = set(det_rows)
    cvpr_ids = set(cvpr_rows)
    if det_ids != cvpr_ids:
        missing_det = sorted(cvpr_ids - det_ids)[:10]
        missing_cvpr = sorted(det_ids - cvpr_ids)[:10]
        raise ValueError(
            "prediction files do not contain identical instance IDs; "
            f"missing from Det2Ver={missing_det}, missing from CVPR-FV={missing_cvpr}"
        )

    joined: List[dict] = []
    for key in sorted(det_ids):
        det = det_rows[key]
        cvpr = cvpr_rows[key]
        det_gold = str(det.get("gold_label", ""))
        cvpr_gold = str(cvpr.get("gold_label", ""))
        if det_gold != cvpr_gold:
            raise ValueError(f"gold-label mismatch for {join_field}={det[join_field]!r}")
        if det_gold not in VALID_LABELS:
            raise ValueError(
                f"invalid gold label {det_gold!r} for {join_field}={det[join_field]!r}"
            )
        det_prediction = str(det.get("prediction", ""))
        cvpr_prediction = str(cvpr.get("prediction", ""))
        if det_prediction not in VALID_LABELS or cvpr_prediction not in VALID_LABELS:
            raise ValueError(
                f"invalid prediction for {join_field}={det[join_field]!r}: "
                f"Det2Ver={det_prediction!r}, CVPR-FV={cvpr_prediction!r}"
            )
        det_claim = str(det.get("claim", ""))
        cvpr_claim = str(cvpr.get("claim", ""))
        if det_claim and cvpr_claim and det_claim != cvpr_claim:
            raise ValueError(f"claim mismatch for {join_field}={det[join_field]!r}")

        det_conflict, det_answers = recompute_conflict(det, "Det2Ver", threshold)
        cvpr_conflict, cvpr_answers = recompute_conflict(cvpr, "CVPR-FV", threshold)
        v = _probability(cvpr, "v", "CVPR-FV")
        joined.append({
            "instance_id": det[join_field],
            "instance_idx_det2ver": det.get("instance_idx"),
            "instance_idx_cvpr_fv": cvpr.get("instance_idx"),
            "claim": cvpr_claim or det_claim,
            "gold_label": det_gold,
            "det_prediction": det_prediction,
            "cvpr_prediction": cvpr_prediction,
            "det_q_true": float(det["q_true"]),
            "det_q_uncertain": float(det["q_uncertain"]),
            "det_q_false": float(det["q_false"]),
            "cvpr_q_true": float(cvpr["q_true"]),
            "cvpr_q_uncertain": float(cvpr["q_uncertain"]),
            "cvpr_q_false": float(cvpr["q_false"]),
            "det_binary_answers_true_uncertain_false": det_answers,
            "cvpr_binary_answers_true_uncertain_false": cvpr_answers,
            "det_conflict": det_conflict,
            "cvpr_conflict": cvpr_conflict,
            "v": v,
            "stratum": stratum_for(v),
        })
    return joined


def analyze_joined(joined: Sequence[dict]) -> List[dict]:
    output: List[dict] = []
    for label, _lower, _upper in STRATA:
        subset = [row for row in joined if row["stratum"] == label]
        n = len(subset)
        det_conflicts = sum(bool(row["det_conflict"]) for row in subset)
        cvpr_conflicts = sum(bool(row["cvpr_conflict"]) for row in subset)
        output.append({
            "stratum": label,
            "n": n,
            "det_conflicts": det_conflicts,
            "cvpr_conflicts": cvpr_conflicts,
            "det_conflict_rate": percent(det_conflicts, n),
            "cvpr_conflict_rate": percent(cvpr_conflicts, n),
            "det_nei_f1": class_f1(subset, "det_prediction"),
            "cvpr_nei_f1": class_f1(subset, "cvpr_prediction"),
        })
    return output


def analyze(
    det_rows: Mapping[str, dict],
    cvpr_rows: Mapping[str, dict],
    join_field: str = "instance_idx",
    threshold: float = 0.5,
) -> List[dict]:
    """Compatibility wrapper used by tests and downstream scripts."""
    return analyze_joined(join_predictions(det_rows, cvpr_rows, join_field, threshold))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(
    results: Sequence[dict],
    total: int,
    join_field: str,
    threshold: float,
    det_sha256: str,
    cvpr_sha256: str,
) -> str:
    lines = [
        "# Reproduced conflict-rate analysis",
        "",
        f"Instances: **{total:,}**; join field: `{join_field}`; Yes threshold: `{threshold:g}`.",
        "A conflict is recomputed when the `(true, uncertain, false)` binary-answer triple matches none of `(Yes, No, No)`, `(No, Yes, No)`, or `(No, No, Yes)`.",
        "",
        f"- Det2Ver predictions SHA-256: `{det_sha256}`",
        f"- CVPR-FV predictions SHA-256: `{cvpr_sha256}`",
        "",
        "| CVP stratum | n | Det2Ver conflicts | CVPR-FV conflicts | Det2Ver NEI-F1 | CVPR-FV NEI-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['stratum']} | {row['n']:,} | "
            f"{row['det_conflicts']:,}/{row['n']:,} ({row['det_conflict_rate']:.1f}%) | "
            f"{row['cvpr_conflicts']:,}/{row['n']:,} ({row['cvpr_conflict_rate']:.1f}%) | "
            f"{row['det_nei_f1']:.3f} | {row['cvpr_nei_f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and audit the FEVER spurious-conflict analysis."
    )
    parser.add_argument("--det2ver-pred", type=Path, required=True)
    parser.add_argument("--cvpr-pred", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--expected-instances",
        type=int,
        help="Fail unless this many joined instances are present (FEVER: 9985).",
    )
    parser.add_argument("--output", type=Path, help="Markdown summary path.")
    parser.add_argument(
        "--details-output", type=Path, help="Auditable per-instance joined JSONL path."
    )
    parser.add_argument("--summary-json", type=Path, help="Machine-readable summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")

    det_raw = load_jsonl(args.det2ver_pred)
    cvpr_raw = load_jsonl(args.cvpr_pred)
    join_field = choose_join_field(det_raw, cvpr_raw)
    det_rows = index_rows(det_raw, join_field, args.det2ver_pred)
    cvpr_rows = index_rows(cvpr_raw, join_field, args.cvpr_pred)
    joined = join_predictions(det_rows, cvpr_rows, join_field, args.threshold)
    if args.expected_instances is not None and len(joined) != args.expected_instances:
        raise ValueError(
            f"joined {len(joined)} instances, expected {args.expected_instances}"
        )

    results = analyze_joined(joined)
    if sum(row["n"] for row in results) != len(joined):
        raise AssertionError("stratum counts do not cover every joined instance")

    det_digest = sha256(args.det2ver_pred)
    cvpr_digest = sha256(args.cvpr_pred)
    markdown = render_markdown(
        results, len(joined), join_field, args.threshold, det_digest, cvpr_digest
    )
    print(markdown, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.details_output:
        write_jsonl(joined, args.details_output)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps({
                "instance_count": len(joined),
                "join_field": join_field,
                "yes_threshold": args.threshold,
                "valid_lookup_rows_true_uncertain_false": sorted(VALID_LOOKUP_ROWS),
                "det2ver_predictions_sha256": det_digest,
                "cvpr_fv_predictions_sha256": cvpr_digest,
                "strata": results,
            }, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

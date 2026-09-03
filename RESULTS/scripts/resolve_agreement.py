#!/usr/bin/env python3
"""Compute agreement statistics from a completed human-evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


LABELS = ("Verifiable", "Unverifiable", "Uncertain")
ANNOTATOR_COLUMNS = ("annotator_1", "annotator_2", "annotator_3")
ALIASES = {
    "v": "Verifiable",
    "verifiable": "Verifiable",
    "u": "Unverifiable",
    "unv": "Unverifiable",
    "unverifiable": "Unverifiable",
    "?": "Uncertain",
    "uncertain": "Uncertain",
}


def normalize_label(value: str, location: str) -> str:
    label = ALIASES.get(str(value).strip().lower())
    if label is None:
        raise ValueError(f"invalid or blank annotation at {location}: {value!r}")
    return label


def cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cohen kappa requires two non-empty, equally sized vectors")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in LABELS)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def fleiss_kappa(ratings: list[list[str]]) -> float:
    if not ratings:
        raise ValueError("Fleiss kappa requires at least one item")
    raters = len(ratings[0])
    if raters < 2 or any(len(row) != raters for row in ratings):
        raise ValueError("every item must contain the same number of raters")

    item_agreement = []
    totals = Counter()
    for row in ratings:
        counts = Counter(row)
        totals.update(row)
        item_agreement.append(
            (sum(count * count for count in counts.values()) - raters)
            / (raters * (raters - 1))
        )
    observed = sum(item_agreement) / len(item_agreement)
    denominator = len(ratings) * raters
    expected = sum((totals[label] / denominator) ** 2 for label in LABELS)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def analyze(rows: list[dict[str, str]]) -> dict:
    ratings: list[list[str]] = []
    pseudo_labels: list[str] = []
    for row_number, row in enumerate(rows, 2):
        ratings.append([
            normalize_label(row.get(column, ""), f"row {row_number}, {column}")
            for column in ANNOTATOR_COLUMNS
        ])
        pseudo_labels.append(
            normalize_label(row.get("pseudo_label", ""), f"row {row_number}, pseudo_label")
        )

    columns = list(zip(*ratings))
    pairwise = [cohen_kappa(list(columns[i]), list(columns[j])) for i, j in combinations(range(3), 2)]
    exact_three_way = sum(len(set(row)) == 1 for row in ratings) / len(ratings)

    resolved: list[tuple[str, str]] = []
    dropped_three_way = 0
    for pseudo, row in zip(pseudo_labels, ratings):
        counts = Counter(row)
        majority, count = counts.most_common(1)[0]
        if count < 2:
            dropped_three_way += 1
            continue
        resolved.append((pseudo, majority))

    correct = sum(pseudo == majority for pseudo, majority in resolved)
    confusion = {
        pseudo: {human: 0 for human in LABELS}
        for pseudo in ("Verifiable", "Unverifiable")
    }
    for pseudo, majority in resolved:
        if pseudo in confusion:
            confusion[pseudo][majority] += 1

    precision = {}
    for pseudo, row in confusion.items():
        total = sum(row.values())
        precision[pseudo] = row[pseudo] / total if total else None

    return {
        "items": len(rows),
        "raters": 3,
        "pairwise_cohen_kappa": pairwise,
        "mean_pairwise_cohen_kappa": sum(pairwise) / len(pairwise),
        "fleiss_kappa": fleiss_kappa(ratings),
        "exact_three_way_agreement": exact_three_way,
        "three_way_splits_dropped": dropped_three_way,
        "majority_resolved_items": len(resolved),
        "pseudo_label_majority_matches": correct,
        "pseudo_label_majority_accuracy": correct / len(resolved) if resolved else None,
        "pseudo_label_precision": precision,
        "confusion": confusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"annotation file is empty: {args.input}")
    result = analyze(rows)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

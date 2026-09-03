#!/usr/bin/env python3
"""Create the stratified 300-claim annotation sheet described in the paper.

The script regenerates pseudo-labels with the released deterministic rule,
then samples an equal number of Verifiable and Unverifiable claims from each
rumor corpus. It deliberately leaves annotator columns blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cvp_pseudo_labeler import label_rd_corpus  # noqa: E402


CORPORA = ("liar", "fnn", "covid")
LABELS = ("Verifiable", "Unverifiable")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def stratified_rows(
    data_root: Path, per_corpus: int, seed: int
) -> Iterable[dict[str, str]]:
    if per_corpus <= 0 or per_corpus % 2:
        raise ValueError("--per-corpus must be a positive even integer")
    per_label = per_corpus // 2
    rng = random.Random(seed)

    for corpus in CORPORA:
        source = data_root / f"{corpus}_train.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"missing rumor corpus: {source}")
        labelled = label_rd_corpus(read_jsonl(source))
        for pseudo_label in LABELS:
            candidates = [row for row in labelled if row["cvp_label"] == pseudo_label]
            if len(candidates) < per_label:
                raise ValueError(
                    f"{corpus} has {len(candidates)} {pseudo_label} rows; "
                    f"{per_label} are required"
                )
            for row in rng.sample(candidates, per_label):
                claim = row.get("claim") or row.get("text") or row.get("statement") or ""
                yield {
                    "corpus": corpus,
                    "source_id": str(row.get("id", "")),
                    "claim": str(claim),
                    "pseudo_label": pseudo_label,
                    "annotator_1": "",
                    "annotator_2": "",
                    "annotator_3": "",
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "rumor"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-corpus", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(stratified_rows(args.data_root, args.per_corpus, args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus", "source_id", "claim", "pseudo_label",
        "annotator_1", "annotator_2", "annotator_3",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

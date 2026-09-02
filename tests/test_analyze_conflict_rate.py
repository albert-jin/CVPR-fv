"""Dependency-free tests for the spurious-conflict analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import analyze_conflict_rate as conflict  # noqa: E402


def prediction(
    instance_id: int,
    instance_idx: int,
    gold: str,
    pred: str,
    q: tuple[float, float, float],
    *,
    v: float | None = None,
) -> dict:
    q_true, q_uncertain, q_false = q
    row = {
        "instance_id": instance_id,
        "instance_idx": instance_idx,
        "claim": f"claim-{instance_id}",
        "gold_label": gold,
        "prediction": pred,
        "q_true": q_true,
        "q_uncertain": q_uncertain,
        "q_false": q_false,
    }
    is_conflict, answers = conflict.recompute_conflict(row, "fixture")
    row["binary_answers_true_uncertain_false"] = answers
    row["lookup_conflict"] = is_conflict
    if v is not None:
        row["v"] = v
    return row


class ConflictAnalysisTests(unittest.TestCase):
    def test_recomputes_conflict_and_joins_by_source_id(self) -> None:
        det_raw = [
            prediction(10, 99, "NEI", "SUPPORT", (0.9, 0.9, 0.9)),
            prediction(20, 98, "SUPPORT", "SUPPORT", (0.9, 0.1, 0.1)),
            prediction(30, 97, "NEI", "NEI", (0.1, 0.9, 0.1)),
        ]
        # Reverse local indices/order: a row-number join would silently fail.
        cvpr_raw = [
            prediction(30, 0, "NEI", "NEI", (0.1, 0.9, 0.1), v=0.8),
            prediction(20, 1, "SUPPORT", "SUPPORT", (0.9, 0.1, 0.1), v=0.4),
            prediction(10, 2, "NEI", "NEI", (0.1, 0.9, 0.1), v=0.2),
        ]
        det = conflict.index_rows(det_raw, "instance_id", Path("det.jsonl"))
        cvpr = conflict.index_rows(cvpr_raw, "instance_id", Path("cvpr.jsonl"))
        joined = conflict.join_predictions(det, cvpr, "instance_id")
        results = conflict.analyze_joined(joined)

        self.assertEqual([row["n"] for row in results], [1, 1, 1])
        self.assertEqual(results[0]["det_conflicts"], 1)
        self.assertEqual(results[0]["cvpr_conflicts"], 0)
        self.assertEqual(results[0]["det_nei_f1"], 0.0)
        self.assertEqual(results[0]["cvpr_nei_f1"], 1.0)

    def test_rejects_inconsistent_exported_conflict_flag(self) -> None:
        row = prediction(1, 0, "SUPPORT", "SUPPORT", (0.9, 0.1, 0.1))
        row["lookup_conflict"] = True
        with self.assertRaisesRegex(ValueError, "disagrees"):
            conflict.recompute_conflict(row, "fixture")

    def test_rejects_mismatched_instance_sets(self) -> None:
        det_row = prediction(1, 0, "SUPPORT", "SUPPORT", (0.9, 0.1, 0.1))
        cvpr_row = prediction(
            2, 0, "SUPPORT", "SUPPORT", (0.9, 0.1, 0.1), v=0.5
        )
        det = conflict.index_rows([det_row], "instance_id", Path("det.jsonl"))
        cvpr = conflict.index_rows([cvpr_row], "instance_id", Path("cvpr.jsonl"))
        with self.assertRaisesRegex(ValueError, "identical instance IDs"):
            conflict.join_predictions(det, cvpr, "instance_id")

    def test_threshold_boundary_is_yes(self) -> None:
        row = prediction(1, 0, "SUPPORT", "SUPPORT", (0.5, 0.1, 0.1))
        is_conflict, answers = conflict.recompute_conflict(row, "fixture")
        self.assertFalse(is_conflict)
        self.assertEqual(answers, [0, 1, 1])


if __name__ == "__main__":
    unittest.main()

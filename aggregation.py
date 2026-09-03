"""Dependency-light inference rules used by CVPR-FV and its ablations."""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Row order is SUPPORT, REFUTE, NEI. Column order is true, uncertain, false.
# Candidate encoding is Yes=0, No=1.
DET2VER_MAP_ROWS = np.array(
    [
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
    ],
    dtype=int,
)


def score_fusion(
    q_true: float,
    q_false: float,
    q_uncertain: float,
    v: float,
    lam: float,
    gamma: float,
    fixed_control: bool = False,
) -> Tuple[int, np.ndarray]:
    """Return the CVPR-FV prediction and normalized logging scores.

    The returned values are decision scores, not posterior probabilities.
    ``fixed_control`` selects the Figure 5 control independent of ``v``.
    """
    inputs = np.asarray([q_true, q_false, q_uncertain, v], dtype=float)
    if not np.isfinite(inputs).all() or ((inputs < 0.0) | (inputs > 1.0)).any():
        raise ValueError("q_true, q_false, q_uncertain, and v must be finite in [0, 1]")
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and non-negative")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite in [0, 1]")

    eps = 1e-12
    raw_scores = np.array(
        [
            q_true * (1 - q_false) * (1 - q_uncertain),
            q_false * (1 - q_true) * (1 - q_uncertain),
            q_uncertain * (1 - q_true) * (1 - q_false),
        ],
        dtype=float,
    )
    decomposition = (raw_scores + eps) / (raw_scores + eps).sum()

    if fixed_control:
        compatibility = np.array([0.25, 0.25, 0.5], dtype=float)
    else:
        r_nei = (1 - v) + gamma * v
        r_support_refute = (1 - gamma) * v / 2.0
        compatibility = np.array(
            [r_support_refute, r_support_refute, r_nei], dtype=float
        )

    log_scores = np.log(decomposition + eps) + lam * np.log(compatibility + eps)
    stable = np.exp(log_scores - log_scores.max())
    decision_scores = stable / stable.sum()
    return int(decision_scores.argmax()), decision_scores


def det2ver_sync(choices_scores: np.ndarray) -> Tuple[int, np.ndarray]:
    """Apply Det2Ver's lookup-then-sequence-score-ranking synchronizer.

    ``choices_scores`` has shape ``(3, 2)`` in `(true, uncertain, false)`
    prefix order and `(Yes, No)` candidate order. Values are
    length-normalized NLLs, so smaller is better.
    """
    scores = np.asarray(choices_scores, dtype=float)
    if scores.shape != (3, 2):
        raise ValueError(
            f"Det2Ver synchronization requires a (3, 2) NLL array, got {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("Det2Ver synchronization received a non-finite NLL")

    binary_predictions = scores.argmin(axis=1)
    matches = np.all(DET2VER_MAP_ROWS == binary_predictions[None, :], axis=1)
    if matches.any():
        prediction = int(np.flatnonzero(matches)[0])
        decision_scores = np.zeros(3, dtype=float)
        decision_scores[prediction] = 1.0
        return prediction, decision_scores

    totals = np.array(
        [
            sum(
                -scores[prefix_idx, target_answer_idx]
                for prefix_idx, target_answer_idx in enumerate(row)
            )
            for row in DET2VER_MAP_ROWS
        ],
        dtype=float,
    )
    stable = np.exp(totals - totals.max())
    decision_scores = stable / stable.sum()
    return int(totals.argmax()), decision_scores

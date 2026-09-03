import unittest

import numpy as np

from aggregation import det2ver_sync, score_fusion


class AggregationTests(unittest.TestCase):
    def test_det2ver_lookup_support(self) -> None:
        prediction, scores = det2ver_sync(
            np.array([[0.1, 2.0], [2.0, 0.1], [2.0, 0.1]])
        )
        self.assertEqual(prediction, 0)
        np.testing.assert_allclose(scores, [1.0, 0.0, 0.0])

    def test_det2ver_conflict_uses_score_fallback(self) -> None:
        # All three binary decisions are Yes, so no lookup row matches.
        prediction, scores = det2ver_sync(
            np.array([[0.1, 1.0], [0.8, 1.0], [0.9, 1.0]])
        )
        self.assertEqual(prediction, 0)
        self.assertAlmostEqual(float(scores.sum()), 1.0)

    def test_low_verifiability_can_raise_nei_score(self) -> None:
        high_v_pred, high_v_scores = score_fusion(0.55, 0.10, 0.45, 0.9, 0.5, 0.1)
        low_v_pred, low_v_scores = score_fusion(0.55, 0.10, 0.45, 0.1, 0.5, 0.1)
        self.assertGreater(low_v_scores[2], high_v_scores[2])
        self.assertIn(high_v_pred, (0, 1, 2))
        self.assertIn(low_v_pred, (0, 1, 2))

    def test_fixed_control_is_independent_of_v(self) -> None:
        _, low = score_fusion(0.6, 0.2, 0.4, 0.0, 2.0, 0.1, fixed_control=True)
        _, high = score_fusion(0.6, 0.2, 0.4, 1.0, 2.0, 0.1, fixed_control=True)
        np.testing.assert_allclose(low, high)


if __name__ == "__main__":
    unittest.main()

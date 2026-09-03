import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "RESULTS" / "scripts" / "resolve_agreement.py"
SPEC = importlib.util.spec_from_file_location("resolve_agreement", SCRIPT)
agreement = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(agreement)


class HumanAgreementTests(unittest.TestCase):
    def test_resolves_majorities_and_drops_three_way_split(self) -> None:
        rows = [
            {
                "pseudo_label": "Verifiable",
                "annotator_1": "V",
                "annotator_2": "Verifiable",
                "annotator_3": "v",
            },
            {
                "pseudo_label": "Unverifiable",
                "annotator_1": "U",
                "annotator_2": "Unverifiable",
                "annotator_3": "unv",
            },
            {
                "pseudo_label": "Verifiable",
                "annotator_1": "Verifiable",
                "annotator_2": "V",
                "annotator_3": "Unverifiable",
            },
            {
                "pseudo_label": "Verifiable",
                "annotator_1": "Verifiable",
                "annotator_2": "Unverifiable",
                "annotator_3": "Uncertain",
            },
        ]
        result = agreement.analyze(rows)
        self.assertEqual(result["items"], 4)
        self.assertEqual(result["three_way_splits_dropped"], 1)
        self.assertEqual(result["majority_resolved_items"], 3)
        self.assertEqual(result["pseudo_label_majority_matches"], 3)
        self.assertAlmostEqual(result["pseudo_label_majority_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

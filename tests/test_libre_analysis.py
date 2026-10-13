import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "libre_analysis", Path(__file__).resolve().parents[1] / "scripts/benchmark/analyze_libre_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExtractionMethodAnalysisTests(unittest.TestCase):
    def analyze(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qc.tsv"
            with path.open("w") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["stem", "extraction_method"])
                writer.writerows(rows)
            return MODULE.analyze_extraction_methods(
                path, {"a": {"f1_c0": "0.6"}},
                {"a": {"f1_c0": "0.4"}, "b": {"f1_c0": "0.8"}},
                [{"sample_key": "b"}])

    def test_missing_predictions_not_imputed(self):
        report = self.analyze([("a", "ccd_template"), ("b", "connectivity_only")])
        self.assertTrue(report["reference_keyset_match"])
        self.assertEqual(len(report["source_sha256"]), 64)
        good = report["methods"]["ccd_template"]
        self.assertAlmostEqual(good["difference"], 0.2)
        rejected = report["methods"]["connectivity_only"]
        self.assertEqual(rejected["n_rejected"], 1)
        self.assertIsNone(rejected["libre_sample_f1"])
        self.assertIsNone(rejected["plc_sample_f1"])

    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.analyze([("a", "ccd_template"), ("a", "ccd_template"), ("b", "connectivity_only")])

    def test_missing_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "keys differ"):
            self.analyze([("a", "ccd_template")])

    def test_missing_method_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing historical"):
            self.analyze([("a", ""), ("b", "connectivity_only")])


class ApprovedExclusionsTests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "policy_id": "test", "approval": {"authority": "user"},
            "failure_signature": {"stage": "qc", "error": "invalid molecule"},
            "datasets": {"demo": {"requested_count": 2, "excluded_sample_keys": ["b"]}},
        }
        self.predictions = {"a": {}}
        self.reference = {"a": {}, "b": {}}
        self.failures = [{"sample_key": "b", "stage": "qc", "error": "invalid molecule"}]

    def accept(self):
        return MODULE.accept_exclusions(self.policy, "demo", self.predictions, self.reference, self.failures)

    def test_accepted_report_does_not_claim_full_coverage(self):
        report = self.accept()
        self.assertTrue(report["complete_under_approved_exclusion_policy"])
        self.assertFalse(report["all_original_rows_have_predictions"])
        self.assertEqual(report["n_matched_pairs_per_model"], 1)
        self.assertEqual(report["excluded_from_both_models"], self.failures)

    def test_unapproved_failure_rejected(self):
        self.failures[0]["sample_key"] = "a"
        with self.assertRaisesRegex(ValueError, "Failures differ"):
            self.accept()

    def test_missing_prediction_rejected(self):
        self.predictions.clear()
        with self.assertRaisesRegex(ValueError, "Matched cohort differs"):
            self.accept()

    def test_unrelated_failure_not_excused(self):
        self.failures[0]["error"] = "OOM"
        with self.assertRaisesRegex(ValueError, "Failure reason differs"):
            self.accept()

    def test_duplicate_exclusions_rejected(self):
        self.policy["datasets"]["demo"]["excluded_sample_keys"].append("b")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.accept()

    def test_changed_population_rejected(self):
        self.reference["c"] = {}
        with self.assertRaisesRegex(ValueError, "reference count mismatch"):
            self.accept()


if __name__ == "__main__":
    unittest.main()

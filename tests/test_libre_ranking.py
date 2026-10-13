import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "libre_ranking", Path(__file__).resolve().parents[1] / "scripts/benchmark/diagnose_libre_ranking.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RankingTests(unittest.TestCase):
    def test_topk_has_exact_budget_and_stable_ties(self):
        p = np.array([.4, .9, .9, .9])
        self.assertEqual(np.flatnonzero(MODULE.top_mask(p, 2)).tolist(), [1, 2])
        self.assertFalse(MODULE.top_mask(p, 0).any())
        self.assertTrue(MODULE.top_mask(p, 10).all())

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.top_mask(np.array([float("nan")]), 1)

    def test_tie_expectation_matches_exhaustive_enumeration(self):
        from itertools import combinations
        p = np.array([.9, .8, .8, .8, .1])
        y = np.array([True, True, False, False, True])
        values = []
        for subset in combinations([1, 2, 3], 2):
            mask = np.zeros(5, bool)
            mask[[0, *subset]] = True
            values.append(MODULE.binary_f1(mask, y))
        r = MODULE.tie_statistics(p, y, 3)
        self.assertAlmostEqual(r["f1_tie_expected"], np.mean(values))
        self.assertEqual(r["f1_tie_min"], min(values))
        self.assertEqual(r["f1_tie_max"], max(values))
        self.assertEqual(r["cutoff_tie"], 1)
        self.assertEqual(r["cutoff_tie_at_one"], 0)
        self.assertEqual(MODULE.tie_statistics(p, y, 0)["f1_tie_expected"], 0.)

    def test_candidate_tie_mean_does_not_choose_gt_best(self):
        ca = np.array([[i, 0, 0] for i in range(6)], float)
        y = np.array([True, True, True, False, False, False])
        r = MODULE.evaluate_candidates(ca, np.ones(6), y, [np.arange(3), np.arange(3, 6)])
        self.assertEqual(r["n_top_score_ties"], 2)
        self.assertEqual(r["selected_f1_tie_mean"], .5)
        self.assertEqual(r["oracle_f1"], 1.)

    def test_budget_does_not_use_targets(self):
        b = MODULE.budgets(31, 0, 25)
        self.assertEqual(b["top20"], 20)
        self.assertEqual(b["top80"], 31)
        self.assertEqual(b["top5pct"], 2)
        self.assertEqual(b["match_plc_count"], 0)

    def test_candidate_selection_is_not_oracle(self):
        ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [20, 0, 0], [21, 0, 0], [22, 0, 0]], float)
        p = np.array([.9, .8, .7, .6, .5, .4])
        proposals = [np.arange(3), np.arange(3, 6)]
        y = np.array([False] * 3 + [True] * 3)
        r = MODULE.evaluate_candidates(ca, p, y, proposals)
        self.assertEqual(r["selected_f1"], 0.)
        self.assertEqual(r["oracle_f1"], 1.)
        self.assertEqual(r["selected_hit_dcc4"], 0.)
        self.assertEqual(r["top3_hit_dcc4"], 1.)
        other = MODULE.evaluate_candidates(ca, p, ~y, proposals)
        self.assertEqual(other["selected_f1"], 1.)

    def test_empty_candidates_and_no_contacts(self):
        r = MODULE.evaluate_candidates(np.zeros((2, 3)), np.ones(2), np.zeros(2, bool), [])
        self.assertIsNone(r["any_hit_dcc4"])
        self.assertIsNone(r["selected_hit_iou025"])
        self.assertEqual(r["selected_f1"], 0.)

    def test_shared_geometry_repeatable(self):
        ca = np.array([[i, 0, 0] for i in range(30)], float)
        a = MODULE.fixed_geometry_candidates(ca)
        b = MODULE.fixed_geometry_candidates(ca)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertLessEqual(len(a), 32)
        self.assertTrue(all(len(x) >= 3 for x in a))

    def test_protein_balance_and_null_denominators(self):
        r = MODULE.aggregate([{"protein_key": "a", "x": 1.}] * 3 +
                             [{"protein_key": "b", "x": 0.}, {"protein_key": "c", "x": None}], ["x"])
        self.assertEqual(r["sample_mean"]["x"], .75)
        self.assertEqual(r["protein_mean"]["x"], .5)
        self.assertEqual(r["metric_n_samples"]["x"], 4)

    def test_paired_difference_and_missing_counterpart(self):
        rows = [{"protein_key": "a", "sample_key": "x", "model": "libre", "f1": .6},
                {"protein_key": "a", "sample_key": "x", "model": "plc", "f1": .4}]
        r = MODULE.paired_difference(rows, "f1", replicates=10)
        self.assertAlmostEqual(r["sample_difference"], .2)
        self.assertAlmostEqual(r["sample_ci95"][0], .2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            MODULE.paired_difference(rows[:1], "f1", replicates=10)

    def test_saved_arrays_require_identical_residue_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = np.array([.8, .2, .1])
            keys = np.array([["A", str(i), ""] for i in range(3)])
            ca = np.array([[i, 0, 0] for i in range(3)], float)
            labels = np.array([0, 1, 2])
            np.savez(root / "plc.npz", probs=np.column_stack((p, (1-p)/2, (1-p)/2)),
                     labels=labels, preds=np.array([0, 2, 2]), ca_coords=ca, residue_keys=keys,
                     protein_key=np.asarray("p"), ligand_key=np.asarray("x"))
            row = dict(n_residues="3", n_pred_contact="1", n_true_contact="1", f1_c0="1",
                       protein_key="p", ligand_key="x")
            for aligned in [True, False]:
                np.savez(root / "libre.npz", probabilities=p, labels=labels == 0, ca_coords=ca,
                         residue_keys=keys if aligned else keys[::-1], smiles=np.asarray("C"))
                if aligned:
                    result = MODULE.load_pair(root / "libre.npz", root / "plc.npz", row, row)
                    self.assertEqual(result[2].tolist(), [True, False, False])
                else:
                    with self.assertRaisesRegex(ValueError, "Residue key mismatch"):
                        MODULE.load_pair(root / "libre.npz", root / "plc.npz", row, row)


if __name__ == "__main__":
    unittest.main()

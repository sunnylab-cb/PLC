import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "libre_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark/run_libre_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiBReBenchmarkTests(unittest.TestCase):
    def test_slurm_launcher_executes_both_tasks(self):
        script = Path(__file__).resolve().parents[1] / "scripts/slurm/run_libre_benchmark.slurm"
        wrapper = '''
function nvidia-smi() { :; }
function fake_python() { printf 'PWD=%s\\n' "$PWD"; printf '%s\\n' "$@"; }
export -f nvidia-smi fake_python
exec bash "$1"
'''
        with tempfile.TemporaryDirectory(prefix="libre shell test ") as root:
            for task, dataset in enumerate(["coach420", "holo4k"]):
                env = dict(os.environ, LIBRE_ROOT=root, LIBRE_PYTHON="fake_python",
                           LIBRE_REFERENCE_ROOT="/fake reference", LIBRE_BENCHMARK_ROOT="/fake benchmark",
                           SLURM_ARRAY_TASK_ID=str(task), SLURM_ARRAY_JOB_ID="1234")
                result = subprocess.run(["bash", "-c", wrapper, "test", str(script)],
                                        env=env, capture_output=True, text=True, check=True)
                args = result.stdout.splitlines()
                self.assertEqual(args[0], "PWD=" + root)
                self.assertEqual(args[args.index("--dataset") + 1], dataset)
                self.assertEqual(args[args.index("--output") + 1],
                                 f"{root}/results/benchmark_1234/{dataset}")
                self.assertEqual(args[args.index("--reference-csv") + 1],
                                 f"/fake reference/{dataset}/val_gate/{dataset}_all_results.csv")

    def test_windows_cover_all_residues(self):
        for length in [1, 1499, 1500, 1501, 1800, 2750, 3001, 7000]:
            counts = np.zeros(length, dtype=int)
            for start, stop in MODULE.windows(length):
                self.assertLessEqual(stop - start, 1500)
                counts[start:stop] += 1
            self.assertTrue((counts > 0).all())

    def test_invalid_windows(self):
        with self.assertRaises(ValueError):
            MODULE.windows(0)
        with self.assertRaises(ValueError):
            MODULE.windows(1501, overlap=1500)

    def test_contact_threshold_is_strict(self):
        result = MODULE.contact_metrics([1, 1, 0], [.6, .8, .7])
        self.assertEqual(result["f1_c0"], .5)
        self.assertEqual(result["n_pred_contact"], 2)

    def test_no_positive_auc_undefined(self):
        result = MODULE.contact_metrics([0, 0], [.1, .9])
        self.assertIsNone(result["auroc_c0"])
        self.assertIsNone(result["ap_c0"])
        self.assertEqual(result["f1_c0"], 0)

    def test_same_protein_assignment_and_conditional_bp(self):
        rows = []
        coords = np.array([[0., 0, 0], [10., 0, 0]])
        for labels in [np.array([True, False]), np.array([False, True])]:
            probabilities = np.array([.9, .1])
            row = {"protein_key": "P", **MODULE.contact_metrics(labels, probabilities)}
            MODULE.centers(row, probabilities, labels, coords)
            rows.append(row)
        MODULE.enrich(rows)
        self.assertEqual([r["pdr_union_success"] for r in rows], [1., 1.])
        self.assertEqual([r["laa_success"] for r in rows], [1., 0.])
        self.assertEqual(rows[0]["bp_f1_c0"], 1.)
        self.assertIsNone(rows[1]["bp_f1_c0"])
        result = MODULE.summarize(rows)
        self.assertEqual(result["sample_mean"]["bp_f1_c0"], 1.)
        self.assertEqual(result["metric_n_samples"]["bp_f1_c0"], 1)

    def test_protein_balancing(self):
        rows = [{"protein_key": "A", "f1_c0": 1., "n_true_contact": 1, "n_pred_contact": 1}]*3
        rows += [{"protein_key": "B", "f1_c0": 0., "n_true_contact": 1, "n_pred_contact": 1}]
        result = MODULE.summarize(rows)
        self.assertEqual(result["sample_mean"]["f1_c0"], .75)
        self.assertEqual(result["protein_mean_of_ligand_mean"]["f1_c0"], .5)


if __name__ == "__main__":
    unittest.main()

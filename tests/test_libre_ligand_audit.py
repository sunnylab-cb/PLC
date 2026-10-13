import importlib.util
from pathlib import Path
import unittest

from rdkit import Chem

SPEC = importlib.util.spec_from_file_location(
    "libre_ligand_audit", Path(__file__).resolve().parents[1] / "scripts/benchmark/audit_libre_ligand_failures.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LigandAuditTests(unittest.TestCase):
    def test_missing_carbon_is_not_equivalent(self):
        report = MODULE.compare_candidate(Chem.MolFromSmiles("CCO"), "CCCO")
        self.assertEqual(report["missing_from_observed"], {"C": 1})
        self.assertFalse(report["same_heavy_connectivity"])

    def test_same_composition_does_not_imply_same_graph(self):
        report = MODULE.compare_candidate(Chem.MolFromSmiles("CCCC"), "CC(C)C")
        self.assertTrue(report["same_heavy_composition"])
        self.assertFalse(report["same_heavy_connectivity"])

    def test_atom_order_does_not_change_connectivity(self):
        report = MODULE.compare_candidate(Chem.MolFromSmiles("CCO"), "OCC")
        self.assertTrue(report["same_heavy_connectivity"])

    def test_bond_order_not_claimed_as_equivalent(self):
        report = MODULE.compare_candidate(Chem.MolFromSmiles("CC"), "C=C")
        self.assertTrue(report["same_heavy_connectivity"])
        self.assertNotIn("chemically_equivalent", report)

    def test_invalid_valence_stays_rejected(self):
        report = MODULE.compare_candidate(Chem.MolFromSmiles("C"), "C(C)(C)(C)(C)C")
        self.assertFalse(report["sanitizes"])
        self.assertFalse(report["same_heavy_connectivity"])


if __name__ == "__main__":
    unittest.main()

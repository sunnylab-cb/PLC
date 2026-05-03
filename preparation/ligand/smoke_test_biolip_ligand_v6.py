#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6-Min ligand feature smoke test.

Runs route/schema/mask/source/status checks on a small set of real BioLiP
examples plus synthetic inorganic, multifragment, and missing-SDF fixtures.
This is intentionally a plain Python script so it can run without pytest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from biolip_ligand_extract import _parse_meta_file, extract_pair
from extract_ligand_features_v5 import extract_ligand_features_v5
from ligand_biolip_types import (
    FEATURE_SCHEMA_VERSION,
    FeatureSource,
    OperationStatus,
    LigandRoute,
    HydrogenPolicy,
    ConformerPolicy,
    LIGAND_TYPE_IDX,
    X_GROUP_FIELDS,
    OP_IDX,
)


GROUP = {name: i for i, name in enumerate(X_GROUP_FIELDS)}
SOURCE_MAX = max(int(v) for v in FeatureSource)
STATUS_SUCCESS = int(OperationStatus.SUCCESS)
STATUS_NOT_APPLICABLE = int(OperationStatus.NOT_APPLICABLE)


REAL_CASES = {
    "BEN": "shared_receptors/3gy7/pairs/3gy7_A/BS02_BEN_A_1/pair_meta.json",
    "ZN": "shared_receptors/8bep/pairs/8bep_L/BS02_ZN_L_700/pair_meta.json",
    "HEM": "shared_receptors/5vcg/pairs/5vcg_A/BS01_HEM_A_601/pair_meta.json",
    "ATP": "shared_receptors/9bn0/pairs/9bn0_A/BS02_ATP_A_4702/pair_meta.json",
    "FAD": "shared_receptors/7fco/pairs/7fco_A/BS01_FAD_A_501/pair_meta.json",
    "NAG": "shared_receptors/1oio/pairs/1oio_A/BS02_NAG_A_202/pair_meta.json",
    "MSE": "shared_receptors/3uso/pairs/3uso_B/BS03_MSE_B_601/pair_meta.json",
}


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _load_pair(meta_path: Path) -> dict:
    pair = _parse_meta_file(meta_path)
    _assert(pair is not None, f"cannot parse pair_meta: {meta_path}")
    return pair


def _extract_to_temp(pair: dict, tmp_root: Path, num_conformers: int = 3) -> dict:
    ok, msg = extract_pair(
        pair,
        tmp_root,
        overwrite=True,
        num_conformers=num_conformers,
        rdkit_threads=1,
        per_pair_timeout=0,
        fallback_no_conformers=True,
    )
    _assert(ok, msg)
    out_files = sorted(tmp_root.rglob("*.pt"))
    _assert(out_files, f"no .pt output for {pair['pdb_id']} {pair['lig_id']}")
    return torch.load(out_files[-1], map_location="cpu", weights_only=False)


def _operation(data: dict, op_name: str) -> int:
    return int(data["operation_status"][OP_IDX[op_name]])


def _group_mask(data: dict, group_name: str) -> torch.Tensor:
    return data["x_group_mask"][:, GROUP[group_name]]


def _group_source(data: dict, group_name: str) -> torch.Tensor:
    return data["x_group_source"][:, GROUP[group_name]]


def _type_bit(data: dict, type_name: str) -> float:
    return float(data["ligand_type"][LIGAND_TYPE_IDX[type_name]])


def _assert_all_finite(name: str, value) -> None:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    _assert(torch.isfinite(tensor).all().item(), f"{name} contains NaN/Inf")


def _assert_binary(name: str, value: torch.Tensor) -> None:
    _assert(torch.logical_or(value == 0, value == 1).all().item(), f"{name} not binary")


def assert_common_schema(data: dict, label: str) -> None:
    x = data["atom_features"]
    n = x.shape[0]

    _assert(data["feature_schema_version"] == FEATURE_SCHEMA_VERSION, f"{label}: bad schema")
    _assert(x.ndim == 2 and x.shape[1] == 109, f"{label}: bad x shape {tuple(x.shape)}")
    _assert(data["edge_index"].shape[0] == 2, f"{label}: bad edge_index")
    _assert(data["edge_attr"].ndim == 2 and data["edge_attr"].shape[1] == 6, f"{label}: bad edge_attr")
    _assert(data["edge_attr"].shape[0] == data["edge_index"].shape[1], f"{label}: edge count mismatch")
    _assert(data["coords"].shape == (n, 3), f"{label}: coords shape mismatch")
    _assert(data["global_features"].shape == (45,), f"{label}: bad global_features")
    _assert(data["shape_features"].shape == (10,), f"{label}: bad shape_features")
    _assert(data["x_group_mask"].shape == (n, 14), f"{label}: bad x_group_mask")
    _assert(data["x_group_source"].shape == (n, 14), f"{label}: bad x_group_source")
    _assert(data["x_atom_extra"].shape == (n, 14), f"{label}: bad x_atom_extra")
    _assert(data["x_atom_extra_mask"].shape == (n, 14), f"{label}: bad x_atom_extra_mask")
    _assert(data["x_atom_extra_source"].shape == (n, 14), f"{label}: bad x_atom_extra_source")
    _assert(data["ligand_type"].shape == (14,), f"{label}: bad ligand_type")
    _assert(data["operation_status"].shape == (12,), f"{label}: bad operation_status")
    _assert(data["fragment_id"].numel() == n, f"{label}: fragment_id must align to current atoms")

    for key in ["atom_features", "coords", "x_atom_extra", "global_features", "shape_features"]:
        _assert_all_finite(f"{label}:{key}", data[key])

    _assert_binary(f"{label}:x_group_mask", data["x_group_mask"])
    _assert_binary(f"{label}:x_atom_extra_mask", data["x_atom_extra_mask"])
    _assert(int(data["x_group_source"].min()) >= 0, f"{label}: bad group source min")
    _assert(int(data["x_group_source"].max()) <= SOURCE_MAX, f"{label}: bad group source max")
    _assert(int(data["x_atom_extra_source"].min()) >= 0, f"{label}: bad atom source min")
    _assert(int(data["x_atom_extra_source"].max()) <= SOURCE_MAX, f"{label}: bad atom source max")

    if data["edge_index"].numel() > 0:
        _assert(int(data["edge_index"].min()) >= 0, f"{label}: negative edge index")
        _assert(int(data["edge_index"].max()) < n, f"{label}: edge index out of range")

    unavailable = {int(FeatureSource.UNAVAILABLE), int(FeatureSource.NOT_APPLICABLE)}
    claiming_success = {
        int(FeatureSource.RDKIT_FULL_SANITIZE),
        int(FeatureSource.RDKIT_PARTIAL_SANITIZE),
        int(FeatureSource.EXPERIMENTAL_COORDS),
    }
    mask = data["x_group_mask"]
    src = data["x_group_source"]
    _assert(
        not (((mask == 0) & torch.isin(src, torch.tensor(list(claiming_success), dtype=src.dtype))).any().item()),
        f"{label}: mask=0 has success-like source",
    )
    _assert(
        not (((mask == 1) & torch.isin(src, torch.tensor(list(unavailable), dtype=src.dtype))).any().item()),
        f"{label}: mask=1 has unavailable/not-applicable source",
    )

    if int(data["conformer_policy"]) != int(ConformerPolicy.FULL_ETKDG_MMFF):
        _assert(float(_group_mask(data, "rmsf_and_flexibility").sum()) == 0.0, f"{label}: non-full route has RMSF mask")
        _assert(_operation(data, "etkdg") != STATUS_SUCCESS, f"{label}: non-full route ETKDG success")
        _assert(_operation(data, "mmff") != STATUS_SUCCESS, f"{label}: non-full route MMFF success")

    for group_name, op_name in [
        ("local_gasteiger_charge", "gasteiger"),
        ("global_morgan_43", "morgan"),
        ("rmsf_and_flexibility", "etkdg"),
        ("sasa_and_lipophilic_sasa", "sasa"),
    ]:
        if _operation(data, op_name) != STATUS_SUCCESS:
            _assert(float(_group_mask(data, group_name).sum()) == 0.0, f"{label}: {group_name} trusted without {op_name} success")


def assert_regular_organic(data: dict, pair: dict) -> None:
    assert_common_schema(data, "BEN")
    _assert(int(data["ligand_route"]) == int(LigandRoute.REGULAR_ORGANIC), "BEN route")
    _assert(_type_bit(data, "regular_organic") == 1.0, "BEN type")
    _assert(int(data["conformer_policy"]) == int(ConformerPolicy.FULL_ETKDG_MMFF), "BEN conformer policy")
    _assert(int(data["hydrogen_policy"]) == int(HydrogenPolicy.V5_ADD_H), "BEN hydrogen policy")
    _assert(_operation(data, "etkdg") == STATUS_SUCCESS, "BEN ETKDG")
    _assert(_operation(data, "mmff") == STATUS_SUCCESS, "BEN MMFF")
    _assert(float(_group_mask(data, "rmsf_and_flexibility").sum()) > 0, "BEN RMSF mask")
    _assert(float(data["feature_completeness_ratio"]) > 0.8, "BEN completeness")

    meta = pair["meta"]
    sdf_name = meta.get("ligand_reference_sdf_file") or meta.get("ligand_reference_mol2_file")
    sdf = pair["pair_dir"] / sdf_name
    # Use sanitize_mode='full' to match extract_pair behaviour for REGULAR_ORGANIC
    v5 = extract_ligand_features_v5(
        str(sdf),
        include_conformers=True,
        num_conformers=3,
        strict=True,
        verbose=False,
        num_threads=1,
        sanitize_mode="full",
    )
    np.testing.assert_allclose(data["global_features"].numpy(), v5["global_features"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(data["shape_features"].numpy(), v5["shape_features"], rtol=1e-5, atol=1e-5)
    _assert(data["atom_features"].shape == torch.as_tensor(v5["atom_features"]).shape, "BEN V5 x shape regression")
    _assert(data["edge_attr"].shape == torch.as_tensor(v5["edge_attr"]).shape, "BEN V5 edge_attr shape regression")


def assert_pure_metal(data: dict, label: str) -> None:
    assert_common_schema(data, label)
    _assert(int(data["ligand_route"]) == int(LigandRoute.PURE_METAL_ION), f"{label}: route")
    _assert(_type_bit(data, "pure_metal_ion") == 1.0, f"{label}: type")
    _assert(data["atom_features"].shape == (1, 109), f"{label}: x shape")
    _assert(data["edge_index"].shape == (2, 0), f"{label}: metal should have empty edge")
    _assert(data["edge_attr"].shape == (0, 6), f"{label}: metal edge_attr")
    _assert(int(data["fragment_count"]) == 1, f"{label}: fragment count")
    _assert(float(data["atom_features"].abs().sum()) > 0, f"{label}: old x all zero")
    _assert(float(_group_mask(data, "shape_3d").sum()) == 0, f"{label}: shape mask")
    for group in ["local_gasteiger_charge", "local_crippen_logp", "local_tpsa_contrib",
                  "rmsf_and_flexibility", "sasa_and_lipophilic_sasa",
                  "global_morgan_43", "shape_3d"]:
        _assert(float(_group_mask(data, group).sum()) == 0, f"{label}: {group} mask")
    for op in ["rdkit_add_h", "etkdg", "mmff", "gasteiger", "morgan"]:
        _assert(_operation(data, op) == STATUS_NOT_APPLICABLE, f"{label}: {op} should be N/A")


def assert_inorganic(data: dict, label: str) -> None:
    assert_common_schema(data, label)
    _assert(int(data["ligand_route"]) == int(LigandRoute.INORGANIC_POLYATOMIC), f"{label}: route")
    _assert(_type_bit(data, "inorganic_polyatomic") == 1.0, f"{label}: type")
    _assert(_type_bit(data, "regular_organic") == 0.0, f"{label}: not regular")
    _assert(int(data["hydrogen_policy"]) == int(HydrogenPolicy.SKIPPED_BY_POLICY), f"{label}: hydrogen")
    for op in ["rdkit_add_h", "etkdg", "mmff", "gasteiger", "morgan"]:
        _assert(_operation(data, op) != STATUS_SUCCESS, f"{label}: {op} unexpectedly success")
    for group in ["local_gasteiger_charge", "local_crippen_logp", "local_tpsa_contrib",
                  "rmsf_and_flexibility", "global_morgan_43"]:
        _assert(float(_group_mask(data, group).sum()) == 0, f"{label}: {group} mask")
    _assert(int(data["fragment_count"]) == 1, f"{label}: fragment count")


def assert_metal_complex(data: dict) -> None:
    assert_common_schema(data, "HEM")
    _assert(int(data["ligand_route"]) == int(LigandRoute.METAL_COMPLEX), "HEM route")
    _assert(_type_bit(data, "metal_complex") == 1.0, "HEM metal_complex")
    _assert(_type_bit(data, "metalloporphyrin") == 1.0, "HEM metalloporphyrin")
    _assert(_type_bit(data, "regular_organic") == 0.0, "HEM not regular")
    _assert(int(data["conformer_policy"]) == int(ConformerPolicy.RAW_COORDS_ONLY), "HEM conformer policy")
    _assert(_operation(data, "etkdg") != STATUS_SUCCESS, "HEM ETKDG")
    _assert(_operation(data, "mmff") != STATUS_SUCCESS, "HEM MMFF")
    _assert(data["coords"].shape[0] == data["atom_features"].shape[0], "HEM coords/x")


def assert_no_conformer_type(data: dict, label: str, type_name: str) -> None:
    assert_common_schema(data, label)
    _assert(int(data["ligand_route"]) == int(LigandRoute.NO_CONFORMER_BIOLOGICAL), f"{label}: route")
    _assert(_type_bit(data, type_name) == 1.0, f"{label}: type {type_name}")
    _assert(int(data["conformer_policy"]) == int(ConformerPolicy.RAW_COORDS_ONLY), f"{label}: conformer")
    _assert(_operation(data, "etkdg") != STATUS_SUCCESS, f"{label}: ETKDG")
    _assert(_operation(data, "mmff") != STATUS_SUCCESS, f"{label}: MMFF")
    _assert(float(_group_mask(data, "rmsf_and_flexibility").sum()) == 0, f"{label}: RMSF mask")


def assert_unusual(data: dict) -> None:
    assert_common_schema(data, "MSE")
    _assert(_type_bit(data, "unusual_element") == 1.0, "MSE unusual element")
    _assert(torch.isfinite(data["x_atom_extra"]).all().item(), "MSE x_atom_extra finite")
    known_mask = data["x_atom_extra_mask"].sum(dim=1) > 0
    _assert(known_mask.any().item(), "MSE should have known periodic-table fields")


def assert_multifragment(data: dict) -> None:
    assert_common_schema(data, "MULTI")
    _assert(int(data["ligand_route"]) == int(LigandRoute.MULTIFRAGMENT_NO_CONFORMER), "MULTI route")
    _assert(_type_bit(data, "multifragment") == 1.0, "MULTI multifragment")
    _assert(_type_bit(data, "regular_organic") == 0.0, "MULTI not regular")
    _assert(int(data["fragment_count"]) > 1, "MULTI fragment_count")
    _assert(data["fragment_id"].shape[0] == data["atom_features"].shape[0], "MULTI fragment_id aligns to x")
    _assert(torch.unique(data["fragment_id"]).numel() == int(data["fragment_count"]), "MULTI fragment ids")
    _assert(float(data["largest_fragment_atom_fraction"]) < 1.0, "MULTI largest fraction")
    _assert(bool(data["fragment_pruned"]) is False, "MULTI fragment_pruned")
    _assert(int(data["conformer_policy"]) == int(ConformerPolicy.RAW_COORDS_ONLY), "MULTI conformer")
    _assert(int(data["hydrogen_policy"]) == int(HydrogenPolicy.EXPLICIT_ONLY), "MULTI hydrogen")
    for op in ["rdkit_add_h", "etkdg", "mmff", "gasteiger", "crippen_logp", "tpsa", "morgan", "sasa", "shape"]:
        _assert(_operation(data, op) == int(OperationStatus.SKIPPED_BY_POLICY), f"MULTI {op}")
    for group in ["local_gasteiger_charge", "local_crippen_logp", "local_tpsa_contrib",
                  "rmsf_and_flexibility", "sasa_and_lipophilic_sasa",
                  "global_morgan_43", "global_mollogp", "global_tpsa", "shape_3d"]:
        _assert(float(_group_mask(data, group).sum()) == 0.0, f"MULTI {group} mask")
        valid_sources = {
            int(FeatureSource.NOT_APPLICABLE),
            int(FeatureSource.UNAVAILABLE),
            int(FeatureSource.FALLBACK_HEURISTIC),
        }
        _assert(set(_group_source(data, group).tolist()).issubset(valid_sources), f"MULTI {group} source")
    _assert(torch.count_nonzero(data["global_features"]).item() == 0, "MULTI global zero")
    _assert(torch.count_nonzero(data["shape_features"]).item() == 0, "MULTI shape zero")


def _write_sdf_from_smiles(smiles: str, path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    _assert(mol is not None, f"bad smiles: {smiles}")
    AllChem.Compute2DCoords(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _make_synthetic_pair(root: Path, pdb_id: str, lig_id: str, smiles: Optional[str], missing: bool = False) -> dict:
    pair_dir = root / "shared_receptors" / pdb_id / "pairs" / f"{pdb_id}_A" / f"BS01_{lig_id}_A_1"
    pair_dir.mkdir(parents=True, exist_ok=True)
    sdf_name = f"{pdb_id}_{lig_id}_ligand_ori.sdf"
    if smiles is not None and not missing:
        _write_sdf_from_smiles(smiles, pair_dir / sdf_name)
    meta = {
        "protein_chain": "A",
        "binding_site_id": "BS01",
        "ligand_id": lig_id,
        "ligand_resseq": "1",
        "ligand_reference_sdf_file": sdf_name,
    }
    (pair_dir / "pair_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return _load_pair(pair_dir / "pair_meta.json")


def _real_pair(biolip_root: Path, label: str) -> Optional[dict]:
    meta = biolip_root / REAL_CASES[label]
    if not meta.exists():
        print(f"[skip] {label}: missing {meta}")
        return None
    return _load_pair(meta)


def run(args: argparse.Namespace) -> None:
    biolip_root = Path(args.biolip_root)
    tmp_root = Path(tempfile.mkdtemp(prefix="lig_v6_smoke_"))
    synthetic_root = tmp_root / "synthetic_biolip"
    out_root = tmp_root / "out"

    try:
        real_tests = [
            ("BEN", assert_regular_organic),
            ("ZN", assert_pure_metal),
            ("HEM", lambda d, p: assert_metal_complex(d)),
            ("ATP", lambda d, p: assert_no_conformer_type(d, "ATP", "nucleotide_cofactor")),
            ("FAD", lambda d, p: assert_no_conformer_type(d, "FAD", "nucleotide_cofactor")),
            ("NAG", lambda d, p: assert_no_conformer_type(d, "NAG", "carbohydrate_glycan")),
            ("MSE", lambda d, p: assert_unusual(d)),
        ]

        for label, checker in real_tests:
            pair = _real_pair(biolip_root, label)
            if pair is None:
                continue
            data = _extract_to_temp(pair, out_root / label, num_conformers=3)
            checker(data, pair)
            print(f"[ok] {label}: route={LigandRoute(int(data['ligand_route'])).name}")

        so4_pair = _make_synthetic_pair(synthetic_root, "syns", "SO4", "[O-]S(=O)(=O)[O-]")
        so4 = _extract_to_temp(so4_pair, out_root / "SO4", num_conformers=3)
        assert_inorganic(so4, "SO4")
        print("[ok] SO4 synthetic inorganic")

        vo4_pair = _make_synthetic_pair(synthetic_root, "synv", "VO4", "[O-][V](=O)([O-])[O-]")
        vo4 = _extract_to_temp(vo4_pair, out_root / "VO4", num_conformers=3)
        assert_inorganic(vo4, "VO4")
        print("[ok] VO4 synthetic metallo-oxyanion")

        multi_pair = _make_synthetic_pair(synthetic_root, "synm", "LIG", "CCO.[Na+]")
        multi = _extract_to_temp(multi_pair, out_root / "MULTI", num_conformers=3)
        assert_multifragment(multi)
        print("[ok] synthetic multifragment")

        missing_pair = _make_synthetic_pair(synthetic_root, "synx", "BAD", None, missing=True)
        ok, msg = extract_pair(missing_pair, out_root / "MISSING", overwrite=True)
        _assert(not ok, "missing SDF should fail")
        _assert("not found" in msg, f"missing SDF message not explicit: {msg}")
        _assert(not list((out_root / "MISSING").rglob("*.pt")), "missing SDF wrote .pt")
        print("[ok] missing-SDF negative case")
    finally:
        if args.keep_tmp:
            print(f"[tmp] kept {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BioLiP ligand V6-Min smoke test")
    parser.add_argument(
        "--biolip-root",
        default="/home/husrcf/Code/ProtBind/BioLip",
        help="BioLiP root containing shared_receptors/",
    )
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary outputs")
    args = parser.parse_args(argv)

    run(args)
    print("[ok] all V6-Min ligand smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

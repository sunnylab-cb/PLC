import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preparation.ligand.biolip_ligand_extract import (
    _filter_pairs_by_stems,
    _load_manifest_stems,
)


def test_manifest_stems_and_exact_pair_filter(tmp_path):
    manifest = tmp_path / "clean.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_stem"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"feature_stem": "1abc_A_BS01_LIG"})
        writer.writerow({"feature_stem": "2def_B_BS02_ATP"})

    pairs = [
        {"pdb_id": "1abc", "chain": "A", "bs_id": "BS01", "lig_id": "LIG"},
        {"pdb_id": "9zzz", "chain": "A", "bs_id": "BS01", "lig_id": "NO"},
    ]
    selected, missing = _filter_pairs_by_stems(pairs, _load_manifest_stems(manifest))

    assert [pair["pdb_id"] for pair in selected] == ["1abc"]
    assert missing == {"2def_B_BS02_ATP"}

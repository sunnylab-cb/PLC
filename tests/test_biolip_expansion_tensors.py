import torch

from scripts.diagnostics.audit_biolip_expansion_tensors import tensor_qc


def fixture():
    row = dict(pdb_id="1abc", protein_chain="A", binding_site_id="BS01", ligand_id="ATP", heavy_atoms_meta="6")
    data = dict(row, atom_features=torch.zeros(6,109), coords=torch.zeros(6,3),
                edge_index=torch.zeros(2,0,dtype=torch.long), edge_attr=torch.zeros(0,6),
                x_group_mask=torch.ones(6,14), feature_completeness_ratio=1.,
                num_heavy_atoms=6, ligand_route=4)
    return data,row


def test_route4_does_not_mean_failed_features():
    data,row=fixture()
    out=tensor_qc(data,row)
    assert out["expanded_route_gate"] and not out["old_route_gate"]
    assert not out["qc_fail_reasons"]


def test_identity_mismatch_fails_closed():
    data,row=fixture()
    data["protein_chain"]="B"
    assert not tensor_qc(data,row)["expanded_route_gate"]


def test_missing_mask_is_not_measurement_zero():
    data,row=fixture()
    data.pop("x_group_mask")
    assert "missing_group_mask" in tensor_qc(data,row)["qc_fail_reasons"]


def test_low_validity_or_nan_not_ready():
    data,row=fixture()
    data["x_group_mask"].zero_()
    data["feature_completeness_ratio"]=0.
    assert not tensor_qc(data,row)["expanded_route_gate"]
    data,row=fixture()
    data["atom_features"][0,0]=float("nan")
    assert not tensor_qc(data,row)["expanded_route_gate"]

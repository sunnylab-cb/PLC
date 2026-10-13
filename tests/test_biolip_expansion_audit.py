from scripts.diagnostics.audit_biolip_expansion import Bucket, policies


def row(**updates):
    value = dict(pdb_id="1abc", protein_chain="A", binding_site_id="BS01", ligand_id="ATP",
        feature_stem="1abc_A_BS01_ATP", heavy_atoms_meta="31", binding_residue_count="3",
        mapping_status="exact_match", mapping_complete="1", route_meta="no_conformer_biological",
        primary_type_meta="nucleotide_cofactor", pdbbind_split="", binding_residues="A1 A2 A3")
    value.update(updates)
    return value


def test_cofactor_requires_type_expansion_not_size_change():
    p = policies(row())
    assert not p["old_metadata_policy"] and not p["route5_remove_upper_cap"]
    assert p["chemistry_6_512_contact3"]


def test_large_organic_route4_not_rescued_by_removing_route5_cap():
    p = policies(row(primary_type_meta="regular_organic", heavy_atoms_meta="70"))
    assert not p["route5_remove_upper_cap"] and p["chemistry_6_512_contact3"]


def test_bad_mapping_stays_excluded():
    assert not any(policies(row(mapping_complete="0")).values())


def test_ions_not_silently_classed_as_small_molecule_expansion():
    p = policies(row(primary_type_meta="pure_metal_ion", route_meta="pure_metal_ion", heavy_atoms_meta="1"))
    assert p["all_types_exact_positive"] and not p["chemistry_3_512_contact1"]


def test_file_presence_and_known_chain_are_not_certification():
    b = Bucket()
    r = row()
    presence = dict(protein=True, fixed=True, esm=True, ligand=False)
    b.add(r, presence, False, True, False, False)
    out = b.report()
    assert out["counts"]["new_known_clean_chain_ligand_missing"] == 1
    assert out["counts"]["new_known_clean_chain_all_assets_present"] == 0
    assert "NOT homology-cleared" in out["site_proxy"]["scope"]


def test_low_overlap_proxy_requires_different_ligand_ids():
    p = dict(protein=True, fixed=True, esm=True, ligand=True)
    b = Bucket()
    b.add(row(), p, False, False, False, False)
    b.add(row(binding_residues="A20 A21 A22"), p, False, False, False, False)
    assert b.report()["site_proxy"]["groups_different_ccd_and_low_overlap"] == 0
    b.add(row(ligand_id="FAD", binding_residues="A20 A21 A22"), p, False, False, False, False)
    assert b.report()["site_proxy"]["groups_different_ccd_and_low_overlap"] == 1

# BioLiP expansion tracker

Updated 2026-10-13. Assessment, tensor/sequence QC and selective extraction
completed. No expanded training allowlist or model training. These proposals
do not replace any existing experiment.

| Run ID | Milestone | Purpose | System / scope | Priority | Status | Evidence / next gate |
| --- | --- | --- | --- | --- | --- | --- |
| 13001698 | M0 | Candidate inventory | Historical metadata, refreshed file presence | MUST | COMPLETED, exit0 | 95s emergency_cpu; hash-verified summary |
| LOCAL_TESTS | M0 | Audit/preparation predicate checks | Nineteen focused tests + bash syntax | MUST | PASSED | No production data or model changes |
| 13003040 | M1 | PDBbind coverage + actual tensors/sequences | 3,333 existing tensors; known TRAIN chain sequence checks | MUST | COMPLETED, exit0, 4m06s | 1,031 pass tensor gate; 1,023 chain keys match; no homology/geometry approval |
| 13003132 | M2 | Selective missing-ligand extraction | 1,390 candidates on verified known TRAIN chains | MUST | COMPLETED, exit0, 1m56s | 1,390/1,390 first-stage QC pass; route3=391, route4=999; zero approved TRAIN additions |
| PROPOSED_RELEASE | M2 | Freeze expanded TRAIN | Current holdouts; connected components + pairwise audit | MUST | NOT SUBMITTED | Clear homologs/dedup before selective extraction |
| PROPOSED_A44 | M3 | Matched control | Original GFGF, current TRAIN, scratch seed44 | MUST | NOT SUBMITTED | Same update budget and sampling/validity rules as B/C |
| PROPOSED_B44 | M3 | Compatible data expansion | Original GFGF + audited organic additions | MUST if meaningful yield | NOT SUBMITTED | B1 release passes; compare strict VAL |
| PROPOSED_C44 | M3 | Chemistry coverage | Same model + accepted broader chemistry | CONDITIONAL | NOT SUBMITTED | Missing-group semantics and new contact visibility verified |
| PROPOSED_SEEDS_EVAL | M4 | Generalization confirmation | Winner/control seeds43/45; frozen held-out evaluation | CONDITIONAL | NOT SUBMITTED | Seed44 screen passes |
| PROPOSED_SIZE_MATCH | B4 | Isolate diversity | Size-matched old/expanded BioLiP | NICE unless claiming C2 | NOT SUBMITTED | Worthwhile gain observed |

See [plan](EXPERIMENT_PLAN.md) and
[assessment](../../docs/biolip_expansion_assessment_20261013.md), with the
[completed coverage/preparation update](../../docs/pdbbind_coverage_and_biolip_expansion_20261013.md).

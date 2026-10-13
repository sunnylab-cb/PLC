# LiBRe transfer tracker

Date: 2026-10-11, approximately17:15 CST. 46 local tests and GPU smoke passed.
All four tasks of array12964718 completed35epochs with exit0. Strict-VAL metrics
below are not TEST results. No run exceeds the unchanged baseline bestF1 .535996.

| ID | Stage | Purpose | Variant | Split | Priority | Status |
|---|---|---|---|---|---|---|
| A01 | Evidence | Inspect original paper, SI and pinned code | Released LiBRe vs frozen PLC | Existing matched benchmark | Required | Complete; architecture report |
| A02 | Evidence | Check real CNC activity and best epochs | FFFF/GFGF training TB | Strict TRAIN/VAL | Required | Complete; CPU log audit |
| A03 | Sanity | Verify one-key attention and loss semantics | Actual functions on synthetic tensors | No experimental data | Required | Complete; CPU probes |
| S01 | M0 | Real B16 data, production trainer/loss, backward, checkpoint reload | All four arms | TRAIN only | Required | 12964495, completed exit0, 4m47s |
| S02 | Dispatch | Verify smoke/provenance before submitting the four-arm array | No model execution | None | Required | 12964516, completed exit0; submitted12964718 |
| R01 | M1 | Explicit interaction at classifier input | plc_product_readout seed44 | Strict merged | P0 | 12964718_0 completed exit0, 35epochs; bestF1 .527161 at e25 |
| R02 | M1 | Equal-parameter non-product control | plc_additive_readout_control seed44 | Strict merged | P0 | 12964718_1 completed exit0, 35epochs; bestF1 .527424 at e32 |
| R03 | M1 | Per-ligand far-negative penalty | plc_cnc_off seed44 | Strict merged | P0 diagnostic | 12964718_2 completed exit0, 35epochs; bestF1 .535429 at e32 |
| R04 | M2 | Sampled contact/far class separation | plc_residue_classmass seed44 | Strict merged | P1 | 12964718_3 completed exit0, 35epochs; bestF1 .531508 at e32 |
| R05 | M3 | Seed confirmation | Surviving arms + controls seeds43/45 | Strict merged | Conditional | Not submitted |
| R06 | M4 | Held-out generalization and ligand use | Frozen selected arms | TEST/COACH/HOLO | Conditional | Not submitted |

Implementation is default-off and checkpoint-compatible. The remote code is an
isolated copy of the frozen GFGF snapshot, with an explicit nine-file overlay.
Training data, original benchmark predictions and pre-existing jobs are unchanged.
Earlier assignment-run history remains in the parent tracker and is not overwritten.
See `docs/libre_transfer_execution_20261010.md` for protocol and deployment details.
See `docs/libre_transfer_results_20261011.md` for the final TB comparison;
no source, stopping policy, jobs or model choices were changed during this check.

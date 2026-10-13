# LiBRe transfer tracker

Date: 2026-10-10. Proposed runs below do not have Slurm IDs.

| ID | Stage | Purpose | Variant | Split | Priority | Status |
|---|---|---|---|---|---|---|
| A01 | Evidence | Inspect original paper, SI and pinned code | Released LiBRe vs frozen PLC | Existing matched benchmark | Required | Complete; architecture report |
| A02 | Evidence | Check real CNC activity and best epochs | FFFF/GFGF training TB | Strict TRAIN/VAL | Required | Complete; CPU log audit |
| A03 | Sanity | Verify one-key attention and loss semantics | Actual functions on synthetic tensors | No experimental data | Required | Complete; CPU probes |
| R01 | M1 | Explicit interaction at classifier input | plc_product_readout seed44 | Strict merged | P0 | Proposed, not implemented/submitted |
| R02 | M1 | Equal-parameter non-product control | plc_additive_readout_control seed44 | Strict merged | P0 | Proposed, not implemented/submitted |
| R03 | M1 | Per-ligand far-negative penalty | plc_cnc_off seed44 | Strict merged | P0 diagnostic | Proposed, not submitted |
| R04 | M2 | Sampled contact/far class separation | plc_residue_classmass seed44 | Strict merged | P1 | Proposed, not implemented/submitted |
| R05 | M3 | Seed confirmation | Surviving arms + controls seeds43/45 | Strict merged | Conditional | Not submitted |
| R06 | M4 | Held-out generalization and ligand use | Frozen selected arms | TEST/COACH/HOLO | Conditional | Not submitted |

Mainline model code, training data, original benchmark predictions and existing
jobs were not changed by this analysis. Earlier assignment-run history remains
in the parent tracker and is not overwritten.

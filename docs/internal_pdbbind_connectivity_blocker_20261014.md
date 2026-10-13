# PDBbind internal split: completed alignments, blocked training

Checked October 14, 2026. User request: train the strongest current mainline
from scratch on the new PDBbind-only dataset; leave the joint task running
and defer further joint processing. No old checkpoint or old split may replace
the requested new independent cohort.

## Actual State

Experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_fixed5_20261013/`.

| Stage | Observed result |
| --- | --- |
| PDBbind job 13013561 | FAILED, exit 1:0, elapsed 53m02s |
| Feature/geometry QC | 19,018 accepted / 19,037 input; 19 excluded |
| Active label rule | Native heavy-atom d_min <5.0 A, >=3 positive residues |
| Exact unique input chains | 9,432 |
| Exhaustive aligned pairs | 44,476,596 / 44,476,596, all 2,224 chunks complete |
| Graph edges | 585,538 |
| Sample connected components | ONE, containing all 19,018 samples |
| Three-way split | Refused; no candidate split certified |
| Fresh cross-set audit | Not started |
| New TRAIN-only normalization | Not started |
| New model training | Not submitted |

The job ran October 14 03:19:26 to 04:12:28, cluster time. Peak batch RSS
reported by Slurm was 15,918,696 KB. The traceback is an intentional
`ValueError: fewer than three separable components; refusing to split a family`
in `assign_components`, not OOM or an incomplete alignment computation.

## Why The Graph Collapsed

The contract implemented by the assistant was:

`global SI >=0.30 OR (local SI >=0.30 AND max(query_coverage,target_coverage)>=0.80)`.

The second branch admits near-universal hubs when an input chain is extremely
short. Covering a one-residue sequence completely is not meaningful evidence
of whole-protein homology. Short observed input chains were not excluded by
the feature QC. Union-find then correctly propagates these edges transitively.

| Example | Observed chain sequence | Length | Graph degree |
| --- | --- | ---: | ---: |
| 1eb1 | P | 1 | 9,282 |
| 5jjm | SD | 2 | 8,673 |
| 2wev | RR | 2 | 8,439 |
| 2uue | RLI | 3 | 5,397 |

There are 18 unique chains shorter than 5 residues, 54 shorter than 10,
163 shorter than 20, and 199 shorter than 30. Of the 585,538 edges, 194,724
meet global SI >=0.30 and 390,814 are admitted only by the local predicate.
The graph diagnosis rebuilt components from the saved edges and sequence
membership; it did not alter the pool or reuse an old split.

## Read-only Alternatives, Not Approved Splits

Whole-complex length filtering below means excluding a complex if ANY of its
input chains is below the stated minimum; no chain is silently ignored.

| Diagnostic rule | Retained complexes | Components | Largest component |
| --- | ---: | ---: | ---: |
| Current global OR covered-local | 19,018 | 1 | 19,018 |
| Global SI only | 19,018 | 1,865 | 6,529 |
| Current rule, all chains >=10 | 18,919 | 1 | 18,919 |
| Current rule, all chains >=20 | 18,515 | 77 | 18,327 |
| Current rule, all chains >=30 | 18,378 | 246 | 17,814 |
| Current rule, all chains >=50 | 18,239 | 574 | 16,381 |

Simply deleting the shortest few complexes does not recover a useful balanced
split under the combined rule. No alternative in this table is certified:
assignment, fresh cross-set audit and TRAIN-only preprocessing still have to run.

Recommended decision for user confirmation: use global pairwise SI <0.30 as
the explicit cross-set guarantee, connected-component assignment, and retain
local SI as a diagnostic rather than a hard edge. This preserves the full QC
pool but does NOT certify the superseded local-SI condition or domain-level
independence. The primary contract must not change without acknowledgement.
The completed all-pairs cache contains every global-threshold edge, so it can
be reused with provenance checks; cross-set SI must still be independently
recomputed under the approved rule.

Remote diagnostic artifact:
`pdbbind_only/connectivity_diagnostic_20261014.json` under the experiment root.
This file records the read-only scenarios and short-chain hub examples.

## Training Recipe And Joint Scope

The selected verified mainline is **V11m13 group_rescompat + GFGF**, seed44,
random initialization, fully unfrozen, co-attention, local54 conditional group
evidence, protein gradient scale 0.6, batch16/BF16, AdamW 1e-4, up to 120 epochs,
and no torch.compile. Keep the original loss, stopping policy and best-F1/
ranking checkpoint utilities. No additional ESM3, distillation or context-noise
probe is promoted from the inconclusive one-seed results.

The actual reference config was read from
`protein_hybrid_gdn_20261006/outputs/protein_hybrid_GFGF_scratch_seed44_v11m13_20261008_202709/config.yaml`
under the HPC experiments directory. It already sets `init_from_checkpoint`
and `init_from_v11m3` to null and both freeze flags false. A new run must replace
its old data/split/label paths with certified new-cohort artifacts, then pass
production-loader and GPU smoke checks. Reading this config does not mean a
new training config or GPU submission is ready.

Joint job 13013562 was RUNNING at the initial accounting check. The user's
clarification explicitly revoked the earlier interpretation of "stop joint":
leave it running, do not advance joint preprocessing/training today. No joint
job or artifact has been modified in this follow-up.

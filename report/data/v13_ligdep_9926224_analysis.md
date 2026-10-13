# V13 Ligand-Dependency Summary: Job 9926224

Parsed from remote `results/ligdep_v13_all_9926224`.

Primary metric: `global_f1_c0`.

| run | real | shuffle | zero | real-shuffle | real-zero | P/R |
|---|---:|---:|---:|---:|---:|---:|
| v13m0_warm_resatom_adapter_only | 0.6157 | 0.5356 | 0.4473 | 0.0801 | 0.1684 | 0.6269 / 0.6049 |
| v13m0_warm_value_enrich_only | 0.6155 | 0.5354 | 0.4473 | 0.0801 | 0.1682 | 0.6282 / 0.6034 |
| v13m0_warm_resatom_adapter_head | 0.6155 | 0.5353 | 0.4478 | 0.0802 | 0.1677 | 0.6283 / 0.6033 |
| v13m0_warm_value_enrich_head | 0.6152 | 0.5355 | 0.4476 | 0.0797 | 0.1676 | 0.6287 / 0.6023 |
| v13m1_coattn_slots16 | 0.6121 | 0.5319 | 0.3373 | 0.0802 | 0.2749 | 0.6333 / 0.5924 |
| v13m0_scratch_coattn_resatom_tri | 0.6090 | 0.5222 | 0.3373 | 0.0868 | 0.2716 | 0.6352 / 0.5848 |
| v13m0_scratch_coattn_resatom | 0.6086 | 0.5276 | 0.3526 | 0.0810 | 0.2560 | 0.6312 / 0.5875 |
| v13m2_coattn_qkreparam | 0.6070 | 0.5199 | 0.3848 | 0.0871 | 0.2223 | 0.6336 / 0.5826 |
| v13m2_noncoattn_qkreparam | 0.6016 | 0.5180 | 0.3486 | 0.0836 | 0.2530 | 0.6279 / 0.5774 |
| v13m1_noncoattn_slots16 | 0.6006 | 0.5208 | 0.2400 | 0.0798 | 0.3605 | 0.6199 / 0.5823 |
| v13m0_scratch_noncoattn_resatom_tri | 0.5990 | 0.5131 | 0.2167 | 0.0859 | 0.3823 | 0.6247 / 0.5753 |
| v13m0_scratch_noncoattn_resatom | 0.5960 | 0.5155 | 0.3000 | 0.0806 | 0.2961 | 0.6250 / 0.5696 |

Reference baseline:

| run | real | shuffle | zero | real-shuffle | real-zero |
|---|---:|---:|---:|---:|---:|
| V11m13 group_rescompat seed44 | 0.6158 | 0.5355 | 0.4474 | 0.0802 | 0.1684 |

## Interpretation

1. Warm-start V13 modules are effectively neutral. All four warm V13 variants match V11m13 group_rescompat within ~0.0006 F1 and preserve the same dependency profile.
2. Scratch V13 variants increase ligand-dependency in some cases, especially co-attn triangle and qk-reparam (`real-shuffle` about 0.087), but lower real F1 by about 0.007-0.009 versus V11m13.
3. Non-coattn variants consistently underperform coattn variants. This keeps co-attn as the stronger interaction path.
4. Scratch variants are more fragile under zero ligand (`real-zero` up to 0.38), indicating stronger ligand coupling but worse protein-side floor.
5. Source split remains consistent with earlier conclusions: graph/local54 dominate; UniMol contributes little in most variants. Scratch/slots can make zero-UniMol drops larger, but this does not translate into higher real F1.

## Current Decision

V13 should not replace V11m13 as the mainline. The useful lesson is architectural: richer interaction modules can increase ligand sensitivity, but without preserving the V11m13 protein/local54 decision geometry they lose real F1. Next low-risk path is to keep V11m13/group_rescompat as base and test protein-side feature fusion such as ESM2-MLC and weak ESM3 prior, not to expand V13 scalar/pair-bias branches further.

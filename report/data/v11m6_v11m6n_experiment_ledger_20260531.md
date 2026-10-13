# V11m6/V11m6n Experiment Ledger

Generated: 2026-05-31 from synced remote logs under `tmp/logs_v11_history_remote/` and local result JSON/TB-derived artifacts.

## Artifact Coverage
- Training log rows parsed: 35
- Ligand-dependency core rows parsed: 14
- Ligand source/feature split rows parsed: 84
- Scale moment rows parsed: 7
- `v11m6n_moment_init_9801410_[0-5]` later completed successfully on 2026-06-01; final rows are reflected below and detailed in `tmp/v11m6n_moment_init_9801410_final.md`.

## Seed Statistics
| set | n | mean_best | sample_std | min_best | max_best | members |
|---|---|---|---|---|---|---|
| grad05_no_gate_seed42_46 | 5 | 0.5853 | 0.0027 | 0.5824 | 0.5885 | grad05_no_gate=0.5849; seed43_grad05_no_gate=0.5824; seed44_grad05_no_gate=0.5885; seed45_grad05_no_gate=0.5830; seed46_grad05_no_gate=0.5875 |
| grad05_no_gate_seed43_46 | 4 | 0.5854 | 0.0031 | 0.5824 | 0.5885 | seed43_grad05_no_gate=0.5824; seed44_grad05_no_gate=0.5885; seed45_grad05_no_gate=0.5830; seed46_grad05_no_gate=0.5875 |
| grad060_no_gate_seed42_46 | 5 | 0.5864 | 0.0018 | 0.5840 | 0.5882 | grad060_no_gate=0.5882; seed43_grad060_no_gate=0.5840; seed44_grad060_no_gate=0.5853; seed45_grad060_no_gate=0.5879; seed46_grad060_no_gate=0.5865 |
| grad060_no_gate_seed43_46 | 4 | 0.5859 | 0.0017 | 0.5840 | 0.5879 | seed43_grad060_no_gate=0.5840; seed44_grad060_no_gate=0.5853; seed45_grad060_no_gate=0.5879; seed46_grad060_no_gate=0.5865 |

## Grad Scale / Probe Summary
| variant | best_f1 | best_epoch | last_f1 | tail5 | status | file |
|---|---|---|---|---|---|---|
| grad035_gate | 0.5831 | 78 | 0.5809 | 0.5811 | done | v11m6n_9796534_2.out |
| grad035_no_gate | 0.5825 | 73 | 0.5805 | 0.5800 | done | v11m6n_9796534_3.out |
| grad05_gate | 0.5831 | 78 | 0.5792 | 0.5796 | done | v11m6n_9796534_0.out |
| grad05_no_gate | 0.5849 | 99 | 0.5848 | 0.5842 | done | v11m6n_9796534_1.out |
| grad055_no_gate | 0.5846 | 94 | 0.5837 | 0.5836 | done | v11m6n_next4_9799266_2.out |
| grad0575_no_gate | 0.5863 | 73 | 0.5841 | 0.5841 | done | v11m6n_g575_probe_9801011.out |
| grad060_no_gate | 0.5882 | 91 | 0.5872 | 0.5868 | done | v11m6n_gsprobe_9799885_0.out |
| grad0625_no_gate | 0.5854 | 90 | 0.5848 | 0.5844 | done | v11m6n_g625_probe_9800610.out |
| grad065_no_gate | 0.5853 | 90 | 0.5847 | 0.5840 | done | v11m6n_next4_9799266_3.out |
| grad070_no_gate | 0.5846 | 73 | 0.5836 | 0.5831 | done | v11m6n_gsprobe_9799885_1.out |

## V11m6 Ablation Summary
| variant | best_f1 | best_epoch | last_f1 | tail5 | status | file |
|---|---|---|---|---|---|---|
| no_gate | 0.5787 | 74 | 0.5778 | 0.5775 | done | v11m6_ablate_9795194_0.out |
| local54_only | 0.5776 | 90 | 0.5762 | 0.5761 | done | v11m6_ablate_9795194_1.out |
| lig_hidden_only | 0.5744 | 88 | 0.5729 | 0.5736 | done | v11m6_ablate_9795194_2.out |
| no_detach_protein | 0.5846 | 95 | 0.5845 | 0.5841 | done | v11m6_ablate_9795194_3.out |
| ? | 0.5776 | 71 | 0.5744 | 0.5754 | done | v11m6_delta_9794691.out |
| ? | 0.4455 | 4 | 0.4281 | 0.3945 | running_or_incomplete | v11m6_delta_9794694.out |

## V11m6n Training Ledger
| family | variant | best_f1 | best_epoch | last_f1 | tail5 | status | file |
|---|---|---|---|---|---|---|---|
| v11m6n | grad05_gate | 0.5831 | 78 | 0.5792 | 0.5796 | done | v11m6n_9796534_0.out |
| v11m6n | grad05_no_gate | 0.5849 | 99 | 0.5848 | 0.5842 | done | v11m6n_9796534_1.out |
| v11m6n | grad035_gate | 0.5831 | 78 | 0.5809 | 0.5811 | done | v11m6n_9796534_2.out |
| v11m6n | grad035_no_gate | 0.5825 | 73 | 0.5805 | 0.5800 | done | v11m6n_9796534_3.out |
| v11m6n_follow | seed45_grad05_no_gate | 0.5830 | 83 | 0.5791 | 0.5778 | done | v11m6n_follow_9800180_0.out |
| v11m6n_follow | seed46_grad05_no_gate | 0.5875 | 79 | 0.5862 | 0.5866 | done | v11m6n_follow_9800180_1.out |
| v11m6n_follow | grad05_tversky075_no_gate | 0.5843 | 91 | 0.5835 | 0.5827 | done | v11m6n_follow_9800180_2.out |
| v11m6n_follow | grad065_tversky075_no_gate | 0.5848 | 91 | 0.5842 | 0.5840 | done | v11m6n_follow_9800180_3.out |
| v11m6n_g575_probe | grad0575_no_gate | 0.5863 | 73 | 0.5841 | 0.5841 | done | v11m6n_g575_probe_9801011.out |
| v11m6n_g60_seeds | seed43_grad060_no_gate | 0.5840 | 91 | 0.5828 | 0.5832 | done | v11m6n_g60_seeds_9800456_0.out |
| v11m6n_g60_seeds | seed44_grad060_no_gate | 0.5853 | 98 | 0.5845 | 0.5844 | done | v11m6n_g60_seeds_9800456_1.out |
| v11m6n_g60_seeds | seed45_grad060_no_gate | 0.5879 | 73 | 0.5867 | 0.5855 | done | v11m6n_g60_seeds_9800456_2.out |
| v11m6n_g60_seeds | seed46_grad060_no_gate | 0.5865 | 77 | 0.5865 | 0.5861 | done | v11m6n_g60_seeds_9800456_3.out |
| v11m6n_g625_probe | grad0625_no_gate | 0.5854 | 90 | 0.5848 | 0.5844 | done | v11m6n_g625_probe_9800610.out |
| v11m6n_gsprobe | grad060_no_gate | 0.5882 | 91 | 0.5872 | 0.5868 | done | v11m6n_gsprobe_9799885_0.out |
| v11m6n_gsprobe | grad070_no_gate | 0.5846 | 73 | 0.5836 | 0.5831 | done | v11m6n_gsprobe_9799885_1.out |
| v11m6n_next4 | seed43_grad05_no_gate | 0.5824 | 97 | 0.5817 | 0.5818 | done | v11m6n_next4_9799266_0.out |
| v11m6n_next4 | seed44_grad05_no_gate | 0.5885 | 99 | 0.5877 | 0.5872 | done | v11m6n_next4_9799266_1.out |
| v11m6n_next4 | grad055_no_gate | 0.5846 | 94 | 0.5837 | 0.5836 | done | v11m6n_next4_9799266_2.out |
| v11m6n_next4 | grad065_no_gate | 0.5853 | 90 | 0.5847 | 0.5840 | done | v11m6n_next4_9799266_3.out |

## Prior / Moment Init Controls
| family | variant | best_f1 | best_epoch | last_f1 | tail5 | status | progress | file |
|---|---|---|---|---|---|---|---|---|
| v11m6n_moment_init | qknorm_a12_g60 | 0.5862 | 73 | 0.5856 | 0.5852 | done | e100/100 | v11m6n_moment_init_9801410_0.out |
| v11m6n_moment_init | qknorm_a14_g60 | 0.5861 | 73 | 0.5852 | 0.5850 | done | e100/100 | v11m6n_moment_init_9801410_1.out |
| v11m6n_moment_init | qknorm_a16_g60 | 0.5867 | 87 | 0.5849 | 0.5849 | done | e100/100 | v11m6n_moment_init_9801410_2.out |
| v11m6n_moment_init | qknorm_a14_tiny1e4_g60 | 0.5844 | 66 | 0.5834 | 0.5833 | done | e100/100 | v11m6n_moment_init_9801410_3.out |
| v11m6n_moment_init | qknorm_a14_tiny3e4_g60 | 0.5875 | 82 | 0.5870 | 0.5862 | done | e100/100 | v11m6n_moment_init_9801410_4.out |
| v11m6n_moment_init | delta_scale025_g60 | 0.5862 | 91 | 0.5858 | 0.5855 | done | e100/100 | v11m6n_moment_init_9801410_5.out |
| v11m6n_prior_init | tiny_final_1e4_g60 | 0.5812 | 98 | 0.5807 | 0.5805 | done | e100 b590/593 | v11m6n_prior_init_9801031_0.out |
| v11m6n_prior_init | tiny_final_3e4_g60 | 0.5855 | 72 | 0.5847 | 0.5842 | done | e100 b590/593 | v11m6n_prior_init_9801031_1.out |
| v11m6n_prior_init | qknorm_g60 | 0.5862 | 78 | 0.5849 | 0.5841 | done | e100 b590/593 | v11m6n_prior_init_9801031_2.out |

## Ligand-Dependency Core Test Results
| run | real | shuffle | zero | real_shuffle_gap | real_zero_gap | precision_real | recall_real | source |
|---|---|---|---|---|---|---|---|---|
| main | 0.5942 | 0.5154 | 0.4225 | 0.0788 | 0.1717 | 0.6110 | 0.5783 | ligdep_v11m6_controls_9796442 |
| no_gate | 0.6008 | 0.5257 | 0.4553 | 0.0751 | 0.1455 | 0.6108 | 0.5912 | ligdep_v11m6_controls_9796442 |
| local54_only | 0.5985 | 0.5206 | 0.4796 | 0.0779 | 0.1189 | 0.6116 | 0.5860 | ligdep_v11m6_controls_9796442_ue |
| no_detach_protein | 0.6052 | 0.5229 | 0.3505 | 0.0823 | 0.2547 | 0.6281 | 0.5839 | ligdep_v11m6_controls_9796442_ue |
| grad05_no_gate | 0.6059 | 0.5222 | 0.4338 | 0.0837 | 0.1721 | 0.6337 | 0.5804 | ligdep_v11m6n_grad05_no_gate_9799215_ue |
| grad060_no_gate | 0.6082 | 0.5269 | 0.4716 | 0.0813 | 0.1366 | 0.6357 | 0.5830 | ligdep_v11m6n_seed44_grad060_9800290_ue |
| seed44_grad05_no_gate | 0.6013 | 0.5210 | 0.3388 | 0.0803 | 0.2625 | 0.6159 | 0.5874 | ligdep_v11m6n_seed44_grad060_9800290_ue |

## Ligand Source/Feature Test Drops
| run | real | zero_graph | zero_graph_drop | zero_local54 | zero_local54_drop | zero_global55 | zero_global55_drop | zero_unimol | zero_unimol_drop | source |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 0.5941 | 0.4102 | 0.1841 | 0.4548 | 0.1394 | 0.5697 | 0.0245 | 0.5930 | 0.0012 | ligdep_v11m6_controls_9796442 |
| no_gate | 0.6008 | 0.4794 | 0.1214 | 0.4897 | 0.1111 | 0.5703 | 0.0305 | 0.5946 | 0.0062 | ligdep_v11m6_controls_9796442 |
| local54_only | 0.5986 | 0.5192 | 0.0793 | 0.5397 | 0.0588 | 0.5717 | 0.0269 | 0.5955 | 0.0030 | ligdep_v11m6_controls_9796442_ue |
| no_detach_protein | 0.6052 | 0.3108 | 0.2944 | 0.3673 | 0.2379 | 0.5750 | 0.0302 | 0.6038 | 0.0014 | ligdep_v11m6_controls_9796442_ue |
| grad05_no_gate | 0.6059 | 0.4421 | 0.1638 | 0.4577 | 0.1482 | 0.5756 | 0.0303 | 0.6015 | 0.0044 | ligdep_v11m6n_grad05_no_gate_9799215_ue |
| grad060_no_gate | 0.6083 | 0.4569 | 0.1514 | 0.4618 | 0.1465 | 0.5790 | 0.0293 | 0.6063 | 0.0019 | ligdep_v11m6n_seed44_grad060_9800290_ue |
| seed44_grad05_no_gate | 0.6014 | 0.4847 | 0.1167 | 0.5331 | 0.0683 | 0.5749 | 0.0265 | 0.5887 | 0.0127 | ligdep_v11m6n_seed44_grad060_9800290_ue |

## Scale Moment Key Rows
| run | model_grad_scale | base_logits_std | effective_delta_std | delta_to_base_abs_ratio_p50 | delta_to_base_abs_ratio_p95 | local_attn_entropy_mean | local_attn_top1_mean | local_score_std | learned_delta_scale_effective | learned_qk_alpha |
|---|---|---|---|---|---|---|---|---|---|---|
| pocket_gvp_fusion_no_detach_protein_v11m6_20260528_010519 | nan | 3.5287 | 4.1640 | 0.7561 | 5.1176 | 2.7643 | 0.2256 | 1.6553 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_seed44_grad05_no_gate_v11m6n_20260529_132503 | 0.5000 | 4.4201 | 3.3311 | 0.5000 | 2.7487 | 2.8401 | 0.1821 | 3.3394 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_grad055_no_gate_v11m6n_20260529_132503 | 0.5500 | 4.1444 | 3.4890 | 0.4778 | 4.4173 | 3.5401 | 0.0562 | 1.2703 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523 | 0.6000 | 4.1359 | 3.5429 | 0.4719 | 4.5871 | 3.5323 | 0.0577 | 1.2755 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_grad05_no_gate_v11m6n_20260528_195453 | 0.5000 | 4.1577 | 3.5425 | 0.4721 | 4.3176 | 3.5656 | 0.0546 | 1.2165 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_grad070_no_gate_v11m6n_20260529_202523 | 0.7000 | 3.9350 | 3.4489 | 0.4907 | 4.9867 | 3.5448 | 0.0556 | 1.2653 | 0.4621 | 11.3137 |
| pocket_gvp_fusion_grad065_no_gate_v11m6n_20260529_132512 | 0.6500 | 4.1155 | 3.4654 | 0.4713 | 4.7492 | 3.5462 | 0.0554 | 1.2562 | 0.4621 | 11.3137 |

## Main Findings
- `grad060_no_gate` remains the best single completed training run by val F1: `0.5882@91`; its test real F1 is `0.6082` with real-shuffle gap `0.0813`.
- Multi-seed mean favors grad060 over grad05: grad060 seed42-46 mean best `0.5864` vs grad05 seed42-46 mean best `0.5853`; overlap exists, so the gain is modest but directionally consistent.
- Ligand-awareness is supported by stable real-shuffle gaps around `0.08` for no_detach/V11m6n variants; this is not just a protein-prior gain.
- Source/feature split consistently points to ligand graph and local54 as the effective signal; UniMol/global55 drops are small in the tested heads.
- Full `no_detach_protein` has strong real F1 but extreme zero sensitivity; V11m6n grad protection keeps ligand gap while reducing zero fragility.
- Prior/moment init has not exceeded `grad060_no_gate`; final best row is `qknorm_a14_tiny3e4_g60` at `0.5875`, with sharper attention risk from TB health checks.

## Open Gaps
- Ligand-dependency eval has not yet been run for moment-init winners; `qknorm_a14_tiny3e4_g60` requires real/shuffle/zero before being considered a candidate.
- `grad060` multi-seed ligand-dependency is only available for the seed42 `grad060_no_gate` checkpoint in this ledger; other seed checkpoints need eval if mean±std test evidence is required.

## Generated CSVs
- `tmp/v11m6_v11m6n_training_ledger.csv`
- `tmp/v11m6_v11m6n_ligdep_core_ledger.csv`
- `tmp/v11m6_v11m6n_ligdep_split_ledger.csv`
- `tmp/v11m6_v11m6n_seed_stats.csv`
- `tmp/v11m6_v11m6n_scale_moment_key.csv`

# Sequence-context control tracker

Updated: 2026-10-29 around20:55 CST. Submitted on U; smoke Priority pending,
full scratch training Dependency pending. No GPU smoke result or new VAL score yet.
Scope: [design and evaluation contract](../../docs/sequence_context_controls_20261029.md).

| Arm | Seed | Start | Status |
| --- | ---: | --- | --- |
| G0_small_global | 44 | Scratch | Existing reference13312384_0; not modified |
| M0_pointwise | 44 | Scratch | Smoke13400657_0 Priority pending; train13400659_0 Dependency pending |
| C0_tcn | 44 | Scratch | Smoke13400657_1 Priority pending; train13400659_1 Dependency pending |
| R0_bigru | 44 | Scratch | Smoke13400657_2 Priority pending; train13400659_2 Dependency pending |

Submission policy: `i64m1tga800u`, one GPU/task, three-task array,8 CPUs and
100G RAM per task. Smoke2h, training7d, training depends on all smoke successes.
No resume, no torch.compile, no new features, no TEST/external inference.
Do not equate submitted/dependency-pending with running or completed.

Local verification:105 tests passed across sequence-context, LocalScan, GDN,
LiBRe-transfer and context-noise suites. This includes FP16/BF16 full-model
checkpoint backward with both local54 interaction and chemistry compatibility
enabled, identical unchanged-module initialization, strict state restoration,
chain/gap/padding isolation, and the TCN receptive-field check. Python compile
and Slurm bash syntax checks passed. GPU/production environment results are
still pending; local optional torch-scatter/torch-sparse CUDA-library warnings
do not establish GPU compatibility. Real GPU smoke remains mandatory.

## Deployment Provenance

- Code snapshot: `b4221db`, pushed to `plc/master` before deployment.
- Root: `/hpc2hdd/home/shuang886/PLC/experiments/sequence_context_controls_u_20261029`.
- Archive: `source_b4221db.tar.gz`, about1.7MiB, code/config/tests/docs only.
  Local/remote SHA256 match:
  `e6afaa28e73a6e2b6a23340921c234a95f3dc7da043027f543e38207ac283bed`.
- Frozen manifest: `prepared/provenance.json`, SHA256
  `d172cd81a06994daec9fd4092f1571a4b3d97263bd2e63b8d0b538178350c295`.
- Full configs compared remotely with the actual running G0 YAML. Data,
  training and source-metadata sections are identical. Model sections differ
  only in context variant/operator/CA-break setting; run names/outputs differ.
- Full production model initialization audited on HPC CPU, seed44. Every
  state-dict tensor outside `protein_context_encoder` matches G0 bitwise;
  RNG states after model creation also match. Both local54 interaction K/V
  and residue-chemistry compatibility are enabled in all three models.
- Frozen counts: TRAIN139158, VAL17395, TEST17395. No dataset copy or rebuild.

| Arm | Total parameters | Front parameters |
| --- | ---: | ---: |
| M0_pointwise | 21,212,230 | 1,054,208 |
| C0_tcn | 21,218,374 | 1,060,352 |
| R0_bigru | 21,279,302 | 1,121,280 |

## Slurm Verification

Smoke array `13400657_[0-2]`; training array `13400659_[0-2]`, submitted at
20:54 CST. `scontrol` confirms `afterok:13400657_*`, invalid-dependency
cancellation enabled, `i64m1tga800u`, one node/task,8 CPUs,100G RAM and
`gres/gpu=1`. Training remains blocked until every smoke passes and the script
verifies each successful smoke JSON against the frozen manifest hash.

At the last check smoke was Priority pending; training was Dependency pending.
Old LocalScan jobs `13312384_[0-3]` remained RUNNING and were not modified.
No GPU peak/throughput estimate or new model score is claimed before smoke.
Full training always starts fresh and does not load the smoke weights.

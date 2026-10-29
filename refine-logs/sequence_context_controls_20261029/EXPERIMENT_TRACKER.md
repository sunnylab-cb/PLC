# Sequence-context control tracker

Updated: 2026-10-29. Implementation and CPU verification; deployment pending.
Scope: [design and evaluation contract](../../docs/sequence_context_controls_20261029.md).

| Arm | Seed | Start | Status |
| --- | ---: | --- | --- |
| G0_small_global | 44 | Scratch | Existing reference13312384_0; not modified |
| M0_pointwise | 44 | Scratch | CPU tests passed; GPU smoke pending |
| C0_tcn | 44 | Scratch | CPU tests passed; GPU smoke pending |
| R0_bigru | 44 | Scratch | CPU tests passed; GPU smoke pending |

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

Code and job provenance will be appended after actual remote verification.

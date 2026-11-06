# Project Instructions

## Project Status

- User paused PLC on 2026-11-06. Do not submit, resume, restart, or expand
  PLC experiments until the user explicitly resumes the project.
- Star smoke `2862` and continuation array `2863_[0-2]` were cancelled before
  starting (Slurm elapsed time zero). Their dependency chain is cancelled too.
- Preserve all checkpoints, data, configurations, source snapshots and logs.
  The earlier Star baseline and all three HPC controls remain stopped.

## Compute Location

- User rule (2026-11-06): all future experiments must run on Star, not HPC/hpc2.
- Do not submit, requeue, restart, or launch training, evaluation, feature
  extraction, or analysis compute on HPC. Do not use HPC as a fallback.
- HPC access is limited to explicitly requested acceptance of existing results
  and migration of existing artifacts. Use direct file transfers, not jobs.
- On Star, keep at most one project task using one GPU at a time, unless the
  user explicitly changes that limit. Respect the shared Slurm allocation.
- If existing HPC jobs are still running at cutover, report their state and
  clarify their disposition; do not silently cancel or restart them.
- The earlier allow-finish exception was revoked by the user on 2026-11-06.
  Jobs `13573439_0`, `13573439_1`, and `13573439_2` were explicitly cancelled
  and verified absent from the queue. No HPC compute exception remains.
  Preserve their artifacts; any authorized continuation must run on Star.

## File Transfers

- User rule (2026-11-02): do not submit CPU or Slurm jobs for file transfers.
- Use direct `rsync`/`scp`; use `tmux` or `nohup` with logs for long transfers,
  subject to the host's login/transfer-node policy.
- Do not turn a transfer into a CPU job for persistence, dependency waiting,
  retries, or routine checksum verification. If host policy prevents direct
  transfers, report the blocker and ask for an alternative.
- This rule applies to future transfers. Do not cancel existing transfers
  merely to change their execution mechanism without user approval.

## CodeGraph

If `.codegraph/` exists at the repository root, use `codegraph_explore` or
`codegraph explore` before grep/find or file reads to locate or understand
code. If it does not exist, skip CodeGraph; indexing is the user's decision.

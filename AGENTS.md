# Project Instructions

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

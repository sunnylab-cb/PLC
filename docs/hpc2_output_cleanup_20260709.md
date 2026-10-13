# HPC2 PLC Output Cleanup Plan - 2026-07-09

## Scope

Remote project root:

```text
/hpc2hdd/home/shuang886/jhspoolers/PLC
```

Local metadata archive already synced:

```text
docs/hpc2_output_metadata_20260709/outputs/
tmp/hpc2_outputs_manifest_20260709.tsv
```

The metadata archive contains `config.yaml` and `splits.txt` for remote output directories. It intentionally excludes checkpoint files and TensorBoard event files.

## Disk Audit

Remote storage usage:

| path | size |
|---|---:|
| `outputs/` | 3.1T |
| `results/` | 294M |
| `logs/` | 195M |
| `tmp/` | 57M |

The cleanup target is therefore `outputs/`. Evaluation artifacts under `results/` are small and should be kept.

Checkpoint inventory under `outputs/`:

| file type | count | size |
|---|---:|---:|
| `checkpoint_epoch_*.pt` | 3843 | 2583.2G |
| `checkpoint_latest.pt` | 222 | 144.2G |
| `best_f1_c0_model.pt` | 204 | 130.5G |
| `best_dice_c0_model.pt` | 204 | 130.5G |
| `best_model.pt` | 222 | 144.2G |
| all `.pt` | 4695 | 3132.8G |

Main point: epoch checkpoints alone account for about 2.52T. Removing them keeps best checkpoints and recovers most space.

## Family-Level Space

| family | dirs | total | epoch checkpoints |
|---|---:|---:|---:|
| `v11m6n_grad` | 36 | 508.7G | 423.4G |
| `v11m8_9` | 32 | 492.8G | 410.1G |
| `v11m10_l54kv` | 18 | 410.3G | 341.3G |
| `other` | 45 | 402.9G | 323.1G |
| `unimol_v11m12` | 20 | 279.9G | 228.5G |
| `mamba` | 10 | 231.4G | 192.4G |
| `backup` | 1 | 231.1G | 204.7G |
| `v13` | 22 | 179.7G | 146.0G |
| `esm2_mlc_exact` | 10 | 163.0G | 130.2G |
| `v12` | 7 | 107.5G | 91.7G |
| `esm3_fullfinetune` | 4 | 50.6G | 35.3G |
| `esm3_scratch` | 4 | 48.7G | 33.4G |
| `group_rescompat` | 3 | 15.6G | 11.7G |
| `esm3_other` | 5 | 15.5G | 11.6G |

## Promising Checkpoints To Keep

Keep at least `best_f1_c0_model.pt`, `config.yaml`, and `splits.txt` for these:

| priority | run family | reason |
|---|---|---|
| P0 | `pocket_gvp_fusion_seed44_esm3_global_prior_scratch_v11m13e3ft_20260702_034855` | Current best single model: test real F1 0.6286, real-shuffle gap 0.0739, HOLO4K sample F1 0.3568, COACH420 F1 0.4960. |
| P0 | `pocket_gvp_fusion_esm3_global_prior_scratch_seed43/45/46_v11m13e3ft_*` | Seed support for ESM3 scratch benchmark means. |
| P1 | `pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114` | Main V11m13 reference baseline: test real F1 0.6158, real-shuffle gap 0.0802. |
| P1 | `pocket_gvp_fusion_seed45_group_rescompat_lr2e4_v11m13_20260614_022638` | Group-rescompat replicate/control. |
| P1 | `pocket_gvp_fusion_*grad060_no_gate_v11m6n*` | Historical grad-scale baseline and ligand-dependency evidence. |
| P2 | `pocket_gvp_fusion_esm3_global_prior_fullfinetune_seed*` and `seed44_esm3_global_prior_fullfinetune_*` | Comparison branch; keep best only, not epoch history. |
| P2 | selected top-5 historical benchmark dirs: `u_mem1b2`, `u_mem4b2`, `all3`, `chem_adapter`, `group_rescompat maxatoms512` | Only needed if reproducing old top-5 benchmark comparisons. |

No current reason to keep epoch checkpoints for any of the above.

## Low-Value / Superseded Branches

These can be reduced to metadata only, or at most `best_f1_c0_model.pt` if a checkpoint is still useful for sanity checks:

| branch | cleanup recommendation | reason |
|---|---|---|
| `outputs/backup/` | delete whole subtree after metadata sync | old V11/V12 backup; 231G, superseded by V11m13/ESM3. |
| `v11m10_l54kv` scans | delete whole dirs or best-only | superseded by V11m13 group-rescompat. |
| `v11m11` Mamba/BiMamba | delete whole dirs or best-only | dependency eval did not justify retaining full checkpoints. |
| `v12*` pair-delta branches | delete whole dirs or best-only | lower than current mainline; historical only. |
| `v13` scalar/pair-bias probes | delete whole dirs or best-only | ligdep summary already archived; did not beat V11m13/ESM3. |
| ESM2 MLC / exact scratch controls | best-only or delete after report | underperformed ESM3 scratch and current baseline. |
| most `v11m8/v11m9/all_light/ligdrop` branches | delete whole dirs or best-only | useful as exploration history, not current mainline. |

## Recommended Cleanup Phases

### Phase 0: metadata only, already done

Synced all remote `config.yaml` and `splits.txt` into:

```text
docs/hpc2_output_metadata_20260709/outputs/
```

Generated full size manifest:

```text
tmp/hpc2_outputs_manifest_20260709.tsv
```

### Phase 1 completion status

Completed on 2026-07-09.

Actions executed:

```text
deleted checkpoint_epoch_*.pt: 3843 files
deleted nested/non-top-level *.pt: 34 files
```

Local audit lists:

```text
tmp/delete_epoch_checkpoints_20260709.txt
tmp/delete_nested_pt_checkpoints_20260709.txt
```

Post-cleanup remote storage:

| path | size |
|---|---:|
| `outputs/` | 529G |
| `results/` | 294M |
| `logs/` | 195M |
| `tmp/` | 57M |

Post-cleanup checkpoint inventory:

| file type | count | size |
|---|---:|---:|
| `checkpoint_epoch_*.pt` | 0 | 0.0G |
| nested/non-top-level `.pt` | 0 | 0.0G |
| `checkpoint_latest.pt` | 206 | 131.8G |
| `best_f1_c0_model.pt` | 203 | 129.9G |
| `best_dice_c0_model.pt` | 203 | 129.9G |
| `best_model.pt` | 206 | 131.8G |
| all `.pt` | 818 | 523.4G |

This means all historical epoch checkpoints and nested backup checkpoints have been removed. Top-level run checkpoints are still preserved, including `best_f1_c0_model.pt`, `best_dice_c0_model.pt`, `best_model.pt`, and `checkpoint_latest.pt`.

### Phase 1: safe cleanup, delete epoch checkpoints

Status: completed.

Expected recovery: about 2.52T.

Dry run:

```bash
ssh hpc2login.hpc.hkust-gz.edu.cn \
  'cd /hpc2hdd/home/shuang886/jhspoolers/PLC &&
   find outputs -type f -name "checkpoint_epoch_*.pt" -mtime +2 -print |
   tee tmp/delete_epoch_checkpoints_20260709.txt |
   wc -l'
```

Execute:

```bash
ssh hpc2login.hpc.hkust-gz.edu.cn \
  'cd /hpc2hdd/home/shuang886/jhspoolers/PLC &&
   find outputs -type f -name "checkpoint_epoch_*.pt" -mtime +2 -delete'
```

This keeps `best_f1_c0_model.pt`, `best_dice_c0_model.pt`, `best_model.pt`, `checkpoint_latest.pt`, config, splits, and TensorBoard logs.

### Phase 2: best-F1-only slimming

Expected additional recovery: roughly 400G if applied broadly.

Delete `checkpoint_latest.pt`, `best_dice_c0_model.pt`, and `best_model.pt` only in directories that already have `best_f1_c0_model.pt`.

Dry run:

```bash
ssh hpc2login.hpc.hkust-gz.edu.cn \
  'cd /hpc2hdd/home/shuang886/jhspoolers/PLC &&
   find outputs -type f \( -name "checkpoint_latest.pt" -o -name "best_dice_c0_model.pt" -o -name "best_model.pt" \) -print |
   while read f; do d=$(dirname "$f"); [ -f "$d/best_f1_c0_model.pt" ] && echo "$f"; done |
   tee tmp/delete_duplicate_best_ckpts_20260709.txt |
   wc -l'
```

Execute:

```bash
ssh hpc2login.hpc.hkust-gz.edu.cn \
  'cd /hpc2hdd/home/shuang886/jhspoolers/PLC &&
   find outputs -type f \( -name "checkpoint_latest.pt" -o -name "best_dice_c0_model.pt" -o -name "best_model.pt" \) -print |
   while read f; do d=$(dirname "$f"); [ -f "$d/best_f1_c0_model.pt" ] && rm -f "$f"; done'
```

This leaves one evaluable checkpoint per run.

### Phase 3: aggressive branch pruning

Only after confirming no pending analysis needs those checkpoints:

```text
delete outputs/backup/
delete v12, v11m10_l54kv, v11m11_mamba, most v13, most ESM2 MLC branches
keep P0/P1 best checkpoints listed above
```

I recommend doing Phase 1 first, then reassessing. Phase 1 alone should reduce `outputs/` from about 3.1T to roughly 0.55T while preserving all best checkpoints.

## Guardrails

- Do not delete `results/`; it is small and contains benchmark/eval summaries.
- Do not delete `report/data/remote_summaries/`; this is the local evidence ledger.
- Do not delete the current ESM3 scratch seed44 best checkpoint unless it has been copied elsewhere.
- If any new training job is running, exclude directories modified in the last 48 hours. Current audit shows no `outputs/` directory modified within the last 2 days, but this should be rechecked immediately before deletion.

## Mirror Before Phase 2

Phase 2 was paused. Before deleting top-level duplicate checkpoints, a non-destructive mirror was started from:

```text
/hpc2hdd/home/shuang886/jhspoolers/PLC
```

to:

```text
/hpc2hdd/home/shuang886/PLC
```

Mirror policy:

```text
rsync without --delete
copy outputs/, results/, and logs/
preserve all currently retained top-level checkpoints
```

Remote background process:

```text
PID: 627563
log: /hpc2hdd/home/shuang886/tmp/plc_outputs_mirror_20260709.log
```

Dry-run estimate before launch:

```text
outputs files to transfer: 1512 regular files
outputs bytes to transfer: 567,247,914,905 bytes
delete count: 0
```

Progress check:

```bash
ssh hpc2login.hpc.hkust-gz.edu.cn \
  'ps -p 627563 -o pid,ppid,stat,etime,cmd;
   tail -20 /hpc2hdd/home/shuang886/tmp/plc_outputs_mirror_20260709.log;
   du -sh /hpc2hdd/home/shuang886/PLC/outputs /hpc2hdd/home/shuang886/PLC/results /hpc2hdd/home/shuang886/PLC/logs 2>/dev/null'
```

Mirror completion:

```text
mirror_done=Thu 09 Jul 2026 01:54:57 PM CST
```

Mirrored directories:

| path | source size | mirror size |
|---|---:|---:|
| `outputs/` | 529G | 529G |
| `results/` | 294M | 357M |
| `logs/` | 195M | 209M |

The apparent `du -sh` difference for `results/` and `logs/` is not a file mismatch; per-file byte verification below is exact. The difference is filesystem block accounting / existing directory metadata after rsync.

Per-file verification from source to mirror:

| subdir | source files | source bytes | mirror files | mirror bytes | missing | mismatched | extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs` | 1512 | 567,247,914,905 | 1512 | 567,247,914,905 | 0 | 0 | 0 |
| `results` | 4249 | 306,515,594 | 4249 | 306,515,594 | 0 | 0 | 0 |
| `logs` | 1192 | 171,405,800 | 1192 | 171,405,800 | 0 | 0 | 0 |

Current status:

```text
Mirror complete and verified.
No deletion of /hpc2hdd/home/shuang886/jhspoolers/PLC has been executed after the mirror.
```

## Source Cleanup After Mirror

User scope correction: do not remove the full source project. Remove only large source-side `outputs`/`output`/`backup` directories under:

```text
/hpc2hdd/home/shuang886/jhspoolers/PLC
```

Deletion launched:

```text
delete_pid=952651
log=/hpc2hdd/home/shuang886/tmp/plc_source_delete_outputs_backup_20260709.log
delete_start=Thu 09 Jul 2026 02:00:30 PM CST
delete_done=Thu 09 Jul 2026 02:00:32 PM CST
```

Deleted source-side directories:

| path | pre-delete size | status |
|---|---:|---|
| `/hpc2hdd/home/shuang886/jhspoolers/PLC/outputs` | 529G | deleted |
| `/hpc2hdd/home/shuang886/jhspoolers/PLC/output` | n/a | missing before cleanup |
| `/hpc2hdd/home/shuang886/jhspoolers/PLC/backup` | 125M | deleted |

Final existence check:

```text
MISSING /hpc2hdd/home/shuang886/jhspoolers/PLC/outputs
MISSING /hpc2hdd/home/shuang886/jhspoolers/PLC/output
MISSING /hpc2hdd/home/shuang886/jhspoolers/PLC/backup
EXISTS  /hpc2hdd/home/shuang886/PLC/outputs
EXISTS  /hpc2hdd/home/shuang886/PLC/results
EXISTS  /hpc2hdd/home/shuang886/PLC/logs
```

Source-side `results/`, `logs/`, `tmp/`, code, configs, data, and reports were not deleted.

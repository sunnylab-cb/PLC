from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path("outputs/pocket_gvp_fusion_v12_20260519_214729/runs")
events = sorted(run.glob("events.out.tfevents.*"))
print("events", [e.name for e in events])
ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
ea.Reload()
tags = set(ea.Tags().get("scalars", []))

keys = [
    "Val/f1_c0",
    "Val/dice_c0",
    "Val/iou_c0",
    "Val/epoch_loss",
    "Val/dice_loss",
    "Val/focal_loss",
    "Val/precision_c0",
    "Val/recall_c0",
    "Train/epoch_loss",
    "Train/epoch_dice_loss",
    "Train/epoch_focal_loss",
    "Health/p2l_gate_abs_mean",
    "Health/p2l_delta_norm",
    "Health/l2p_delta_norm",
    "Gradients/grad_norm_ligand_encoder",
    "Gradients/grad_norm_co_attn_blocks",
    "Gradients/grad_norm_esm_proj",
    "Gradients/grad_norm_gvp_proj",
    "Health/unimol_atom_delta_norm",
    "Health/unimol_ligand_alpha",
]

for key in keys:
    if key not in tags:
        continue
    vals = ea.Scalars(key)
    if not vals:
        continue
    is_metric = any(s in key for s in ["f1", "dice_c0", "iou", "precision", "recall"])
    is_epoch = key.startswith("Val/") or key.startswith("Train/epoch")
    if is_epoch:
        best = max(vals, key=lambda x: x.value) if is_metric else min(vals, key=lambda x: x.value)
        print(f"{key}: last e{vals[-1].step}={vals[-1].value:.6g}, best e{best.step}={best.value:.6g}")
        curve = " ".join(f"e{v.step}:{v.value:.4g}" for v in vals)
        print(f"  curve: {curve}")
    else:
        sample = [vals[0], vals[len(vals) // 2], vals[-1]]
        print(
            f"{key}: first={sample[0].value:.6g}@{sample[0].step}, "
            f"mid={sample[1].value:.6g}@{sample[1].step}, "
            f"last={sample[2].value:.6g}@{sample[2].step}"
        )

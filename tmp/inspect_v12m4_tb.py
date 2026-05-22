from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    run_dir = Path("outputs/hpc_sync/pocket_gvp_fusion_pair_aux_old128_v12m4_20260522_184834")
    event_file = sorted((run_dir / "runs").glob("events.out.tfevents.*"))[-1]
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    keys = [
        "Val/f1_c0",
        "Val/precision_c0",
        "Val/recall_c0",
        "Val/pair_f1_c0",
        "Val/pair_precision_c0",
        "Val/pair_recall_c0",
        "Val/epoch_loss",
        "Val/focal_loss",
        "Val/dice_loss",
        "Train/epoch_pair_aux_loss",
        "Train/epoch_pair_aux_weight",
        "Health/pair_delta_gate",
        "Health/pair_score_topk_mean",
        "Health/pair_score_max_mean",
        "Health/pair_entropy_mean",
        "Gradients/grad_norm_pair_delta_classifier",
        "Gradients/grad_norm_pair_evidence",
        "Gradients/grad_norm_pair_int_pool_proj",
        "Gradients/grad_norm_interaction_tokens",
        "Gradients/grad_norm_interaction_blocks",
        "Gradients/grad_norm_esm_proj",
        "Gradients/grad_norm_gvp_proj",
        "Gradients/grad_norm_ligand_encoder",
    ]
    epochs = [0, 5, 9, 10, 15, 18, 20, 22, 26, 30]
    for key in keys:
        if key not in tags:
            continue
        vals = ea.Scalars(key)
        by_step = {v.step: v.value for v in vals}
        print(f"\n{key}")
        print(" ".join(f"e{e}:{by_step[e]:.6g}" for e in epochs if e in by_step))
        if not key.startswith("Val/") and not key.startswith("Train/epoch"):
            first = vals[0]
            mid = vals[len(vals) // 2]
            last = vals[-1]
            print(
                f"first={first.value:.6g}@{first.step} "
                f"mid={mid.value:.6g}@{mid.step} "
                f"last={last.value:.6g}@{last.step}"
            )
        if key.startswith("Val/"):
            best = max(vals, key=lambda v: v.value) if any(
                token in key for token in ("f1", "dice", "iou", "precision", "recall")
            ) else min(vals, key=lambda v: v.value)
            last = vals[-1]
            print(f"summary: last=e{last.step}:{last.value:.6g} best=e{best.step}:{best.value:.6g}")


if __name__ == "__main__":
    main()

"""
Confident Negative Clipping (CNC) — per-residue loss weight for noisy c_2 labels.

Problem:
    In PDBbind, the same protein can appear with multiple ligands in separate
    entries.  A residue labeled c_2 ("far") for ligand L1 may actually be a
    binding residue for a different ligand L2.  The c_2 label is correct
    *per-ligand* but unreliable as a global negative — it is a noisy negative
    in the Positive-Unlabeled (PU) learning sense.

Solution:
    When the model predicts c_0 or c_1 with high confidence for a residue
    labeled c_2, reduce the loss weight for that residue instead of fully
    penalising the model.  This lets the model discover potential binding-
    capable regions without being punished for correct structural insights.

    Positive labels (c_0, c_1) are always trusted and receive full loss weight.

Theory:
    - Confident Learning (Northcutt et al., JAIR 2021): per-class thresholds
      based on mean self-confidence to identify label errors.
    - Bootstrap Loss (Reed et al., 2015): mix label and model prediction in
      the training target to soften noisy supervision.

Usage:
    weights = compute_cnc_weights(logits, labels, mask, config, epoch)
    # Apply weights element-wise to per-residue loss before reduction.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_cnc_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    clip_thresh: float = 0.85,
    clip_ratio: float = 0.1,
    warmup_epochs: int = 5,
    current_epoch: int = 0,
    far_class: int = 2,
    max_clip_frac: float = 0.3,
) -> torch.Tensor:
    """
    Compute per-residue loss weights for Confident Negative Clipping.

    For residues labeled as ``far_class`` where the model's total probability
    on positive classes (all classes except ``far_class``) exceeds
    ``clip_thresh``, the weight is reduced to ``clip_ratio``.  All other
    residues (including all positive-labeled residues) keep weight 1.0.

    Clipping is disabled during warmup (first ``warmup_epochs`` epochs) so
    the model can learn basic patterns before being allowed to override
    negative labels.

    Args:
        logits:         [B, L, C]  raw model output (pre-softmax).
        labels:         [B, L]     ground-truth class indices.
        mask:           [B, L]     True/1 for valid residues, 0 for padding.
        clip_thresh:    Minimum total positive probability to trigger
                        clipping (recommended 0.85–0.9).
        clip_ratio:     Weight applied to clipped residues (0 = ignore,
                        1 = no effect).  Recommended 0.05–0.1 to keep a
                        weak constraint and prevent model collapse.
        warmup_epochs:  Number of epochs with normal loss before CNC
                        activates.
        current_epoch:  Current training epoch (0-indexed).
        far_class:      Class index for the "far" / negative label.
        max_clip_frac:  Maximum fraction of valid residues per sample that
                        can be clipped.  Prevents runaway clipping when the
                        model becomes overconfident.  Set to 1.0 to disable.

    Returns:
        weights:  [B, L]  per-residue loss multiplier (1.0 or ``clip_ratio``).
    """
    B, L = labels.shape
    device = logits.device

    weights = torch.ones(B, L, device=device, dtype=logits.dtype)

    # During warmup, return uniform weights — model predictions are unreliable
    if current_epoch < warmup_epochs:
        return weights

    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)                      # [B, L, C]

        # Total probability on positive classes (everything except far_class)
        p_positive = 1.0 - probs[:, :, far_class]              # [B, L]

        is_far = (labels == far_class)                          # [B, L]
        is_valid = mask.bool()                                  # [B, L]
        confident_positive = (p_positive >= clip_thresh)        # [B, L]

        # Candidate residues for clipping
        clip_candidates = is_far & confident_positive & is_valid  # [B, L]

        # ── Per-sample cap: at most max_clip_frac of valid residues ──
        if max_clip_frac < 1.0:
            n_valid = is_valid.float().sum(dim=1)               # [B]
            max_clip_count = (n_valid * max_clip_frac).long()   # [B]
            n_candidates = clip_candidates.float().sum(dim=1).long()  # [B]

            # If too many candidates, keep only the top-k most confident
            needs_cap = n_candidates > max_clip_count            # [B]
            if needs_cap.any():
                # For samples that exceed the cap, keep top-k by p_positive
                for b in range(B):
                    if not needs_cap[b]:
                        continue
                    cand_idx = clip_candidates[b].nonzero(as_tuple=True)[0]
                    if len(cand_idx) == 0:
                        continue
                    k = max_clip_count[b].item()
                    if k == 0:
                        clip_candidates[b] = False
                        continue
                    cand_probs = p_positive[b, cand_idx]
                    _, topk_local = cand_probs.topk(k)
                    keep_mask = torch.zeros_like(clip_candidates[b])
                    keep_mask[cand_idx[topk_local]] = True
                    clip_candidates[b] = keep_mask

        weights[clip_candidates] = clip_ratio

    return weights


def compute_cnc_stats(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Compute statistics about CNC clipping for logging / TensorBoard.

    Args:
        weights:  [B, L]  output of ``compute_cnc_weights``.
        mask:     [B, L]  validity mask.

    Returns:
        Dictionary with keys:
          - ``n_clipped``:    total residues clipped in this batch
          - ``n_valid``:      total valid residues
          - ``clip_fraction``: fraction of valid residues that were clipped
    """
    is_valid = mask.bool()
    is_clipped = (weights < 1.0) & is_valid
    n_clipped = is_clipped.sum().item()
    n_valid = is_valid.sum().item()
    return {
        "n_clipped": n_clipped,
        "n_valid": n_valid,
        "clip_fraction": n_clipped / max(n_valid, 1),
    }

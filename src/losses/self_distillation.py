"""TRAIN-only, residue-aligned offline self-distillation (no teacher module)."""

from __future__ import annotations

import hashlib
import json
import math
from types import MethodType

import torch
import torch.nn.functional as F


def tensor_digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous().numpy()
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scheduled_weight(epoch, weight, warmup_epochs=5, ramp_epochs=5):
    if epoch < 0 or not math.isfinite(weight) or weight < 0 or warmup_epochs < 0 or ramp_epochs < 1:
        raise ValueError("Invalid KD schedule")
    return weight * min(1.0, max(0.0, (epoch + 1 - warmup_epochs) / ramp_epochs))


class TeacherCache:
    """Validate the complete TRAIN manifest, then fail closed on each residue axis."""

    def __init__(self, payload, config, splits, expected):
        if payload.get("format_version") != 1 or payload.get("split") != "train":
            raise ValueError("Expected a version-1 TRAIN-only teacher cache")
        for key in ("split_sha256", "label_sha256"):
            if payload.get(key) != expected[key]:
                raise ValueError(f"Teacher {key} mismatch")
        teacher = payload["teacher"]
        if teacher["checkpoint_sha256"] != expected["teacher_checkpoint_sha256"]:
            raise ValueError("Wrong teacher checkpoint")
        if teacher["data_config"] != config["data"]:
            raise ValueError("Teacher/student data preprocessing or cropping mismatch")
        if teacher["model_config_sha256"] != object_digest(config["model"]):
            raise ValueError("Teacher/student architecture mismatch")
        self.rows = {}
        for row in payload["rows"]:
            key, z, y = row["sample_id"], row["logits"], row["labels"]
            if key in self.rows:
                raise ValueError(f"Duplicate teacher row: {key}")
            if (z.device.type != "cpu" or y.device.type != "cpu" or z.requires_grad
                    or y.dtype != torch.int64 or not len(y) or z.shape != (len(y), 3)
                    or not torch.isfinite(z).all() or not ((y >= 0) & (y < 3)).all()):
                raise ValueError(f"Invalid teacher logits/labels: {key}")
            digest = row.get("residue_axis_sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Missing/invalid residue axis: {key}")
            self.rows[key] = row
        train = splits["train"]
        if (len(train) != len(set(train)) or set(self.rows) != set(train)
                or len(self.rows) != payload["n_rows"]
                or set(train) & (set(splits["val"]) | set(splits["test"]))):
            raise ValueError("Teacher coverage is not exactly the disjoint TRAIN split")

    @classmethod
    def load(cls, path, config, splits, expected):
        return cls(torch.load(path, map_location="cpu", weights_only=False), config, splits, expected)

    def aligned_logits(self, batch, device):
        mask = batch["protein_mask"].bool().cpu()
        labels = batch["labels"].cpu().long()
        ids = batch["pdb_ids"]
        if len(ids) != mask.size(0) or len(set(ids)) != len(ids):
            raise ValueError("Invalid/duplicate TRAIN batch IDs")
        padded = torch.zeros((*mask.shape, 3), dtype=torch.float32)
        for i, key in enumerate(ids):
            if key not in self.rows:
                raise ValueError(f"No TRAIN teacher row: {key}")
            row, valid = self.rows[key], mask[i]
            y = labels[i][valid]
            if not torch.equal(y, row["labels"]):
                raise ValueError(f"Teacher/student label or length mismatch: {key}")
            digest = tensor_digest(batch["protein_coords"][i][valid], batch["chain_ids"][i][valid], y)
            if digest != row["residue_axis_sha256"]:
                raise ValueError(f"Teacher/student residue axis mismatch: {key}")
            padded[i, valid] = row["logits"]
        return padded.to(device, non_blocking=True)


def residue_kl(student, teacher, mask, eligible, temperature, labels=None):
    """T^2 KL(teacher||student); row means, with dropped rows contributing zero.

    Mask before softmax: padded or ligand-dropped logits cannot produce NaN
    gradients. The denominator is the full batch, NOT just surviving rows.
    """
    if not math.isfinite(temperature) or temperature <= 0 or student.shape != teacher.shape:
        raise ValueError("Invalid KD temperature/logit shapes")
    if student.ndim != 3 or student.shape[-1] != 3 or mask.shape != student.shape[:2]:
        raise ValueError("Expected [B,L,3] logits and [B,L] mask")
    if eligible.shape != (student.size(0),) or student.size(0) == 0:
        raise ValueError("Expected one ligand-dropout eligibility flag per row")
    valid = mask.bool() & eligible.bool()[:, None]
    row_ids = valid.nonzero(as_tuple=True)[0]
    with torch.autocast(student.device.type, enabled=False):
        s = student[valid].float()
        t = teacher.detach()[valid].float()
        if not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise ValueError("Non-finite valid-residue KD logits")
        logp = F.log_softmax(t / temperature, dim=-1)
        per_residue = (logp.exp() * (logp - F.log_softmax(s / temperature, dim=-1))).sum(-1)
        per_residue = per_residue * temperature ** 2
        counts = mask.bool().sum(-1).clamp_min(1)
        weights = 1.0 / (counts[row_ids].float() * student.size(0))
        terms = per_residue * weights
        loss = terms.sum()
        stats = {"eligible_rows": eligible.sum().detach(), "dropped_rows": (~eligible.bool()).sum().detach(),
                 "eligible_residues": valid.sum().detach(), "raw_T2_loss": loss.detach()}
        if labels is not None:
            y = labels[valid]
            for c in range(3):
                stats[f"true_class{c}_contribution"] = terms[y == c].sum().detach()
                stats[f"true_class{c}_count"] = (y == c).sum().detach()
    return loss, stats


def tracked_source_dropout(self, hl_src, local54):
    """Same draws/order/arithmetic as V11m8; only record the actual keep events."""
    if not self.training:
        return hl_src, local54
    batch_size, device = hl_src.size(0), hl_src.device
    eligible = torch.ones(batch_size, dtype=torch.bool, device=device)
    if self.ligand_source_dropout_p > 0.0:
        keep = (torch.rand(batch_size, 1, 1, device=device) >= self.ligand_source_dropout_p).to(dtype=hl_src.dtype)
        hl_src = hl_src * keep
        local54 = local54 * keep.to(dtype=local54.dtype)
        eligible &= keep[:, 0, 0].bool()
    if self.ligand_dropout_hidden_p > 0.0:
        keep = (torch.rand(batch_size, 1, 1, device=device) >= self.ligand_dropout_hidden_p).to(dtype=hl_src.dtype)
        hl_src = hl_src * keep
        eligible &= keep[:, 0, 0].bool()
    if self.ligand_dropout_local54_p > 0.0:
        keep = (torch.rand(batch_size, 1, 1, device=device) >= self.ligand_dropout_local54_p).to(dtype=local54.dtype)
        local54 = local54 * keep
        eligible &= keep[:, 0, 0].bool()
    self._sd_ligand_keep = eligible
    self._sd_dropout_calls += 1
    return hl_src, local54


def reset_dropout_tracking(model, args):
    model._sd_ligand_keep = None
    model._sd_dropout_calls = 0


def attach_distillation(model, cache, options):
    if hasattr(model, "_sd_cache"):
        raise RuntimeError("Self-distillation already attached")
    model._sd_cache, model._sd_options = cache, dict(options)
    model._apply_source_dropout = MethodType(tracked_source_dropout, model)
    reset_dropout_tracking(model, ())
    model.register_forward_pre_hook(reset_dropout_tracking)


def add_self_distillation(model, logits, batch, labels, mask, hard_loss, epoch, writer, step, log):
    model = getattr(model, "module", model)
    if not model.training:
        raise RuntimeError("KD must never run in validation/test")
    opts = model._sd_options
    weight = scheduled_weight(epoch, opts["weight"], opts["warmup_epochs"], opts["ramp_epochs"])
    if writer and log:
        writer.add_scalar("KD/lambda", weight, step)
        writer.add_scalar("KD/hard_plus_original_aux", hard_loss.detach(), step)
    if weight == 0:
        return hard_loss
    if model._sd_dropout_calls != 1 or model._sd_ligand_keep is None:
        raise RuntimeError("Missing or repeated ligand-source dropout tracking")
    teacher = model._sd_cache.aligned_logits(batch, logits.device)
    loss, stats = residue_kl(logits, teacher, mask, model._sd_ligand_keep, opts["temperature"], labels)
    if writer and log:
        for key, value in {**stats, "weighted_loss": weight * loss.detach()}.items():
            writer.add_scalar("KD/" + key, value, step)
    return hard_loss + weight * loss

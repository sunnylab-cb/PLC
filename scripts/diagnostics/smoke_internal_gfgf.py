#!/usr/bin/env python3
"""One-GPU scratch GFGF smoke using the production BF16 loss and FP16 validation."""

import argparse
import gc
import inspect
import json
import math
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts/train"))
import train_v11m13 as trainer
from src.data.collate import prepare_model_inputs_v11, pocket_collate_fn_v10
from src.data.pocket_dataset_v8 import PocketDatasetV8
from prepare_internal_gfgf import verify_prepared, split_members, source_smoke_ids
from prepare_libre_transfer import sha256
from smoke_libre_transfer import CheckedAdamW, ScalarRecorder, training_batch
from smoke_protein_hybrid import kernel_check
from strict_assignment_common import read_strict_splits
from prepare_internal_codataset_sequences import save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, type=Path)
    args = ap.parse_args()
    root = args.experiment
    manifest = verify_prepared(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm GPU allocation")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    report = dict(passed=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  provenance_sha256=sha256(root/"prepared/provenance.json"), kernel=kernel_check())
    gc.collect()
    torch.cuda.empty_cache()
    cfg = yaml.safe_load((root/"prepared/GFGF.yaml").read_text())
    torch.manual_seed(44)
    random.seed(44)
    np.random.seed(44)
    groups = read_strict_splits(cfg["data"]["split_file"])
    joint = manifest.get("cohort") == "pdbbind_biolip_joint"
    members = (split_members(root/"split/candidate_split.tsv", groups, manifest["cohort"])
               if joint else None)
    train_ids = source_smoke_ids(groups, members, "train", cfg["training"]["batch_size"]) if joint else None
    batch = training_batch(cfg, ids=train_ids)
    model = trainer.create_model(cfg, device)
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Unexpected frozen parameters in scratch model")
    t = cfg["training"]
    optimizer = CheckedAdamW(model.parameters(), lr=t["learning_rate"],
                             weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
    writer = ScalarRecorder()
    reserved = {"model", "dataloader", "optimizer", "device", "epoch", "class_weights"}
    kwargs = {k:v for k,v in t.items() if k in inspect.signature(trainer.base.train_epoch).parameters and k not in reserved}
    kwargs.update(gamma=t["focal_gamma"], writer=writer, log_every=1, scaler=None)
    kwargs.update({"cnc_"+k:v for k,v in t["cnc"].items()})
    kwargs.update({"dist_aux_"+k:v for k,v in t["dist_aux"].items()})
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    metrics = trainer.base.train_epoch(model=model, dataloader=[batch]*3, optimizer=optimizer,
        device=device, epoch=6, **{**kwargs, "class_weights":torch.tensor(t["class_weights"], device=device)})
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("Non-finite smoke loss/metrics")
    training_peak = torch.cuda.max_memory_allocated()/2**30
    # Separate real VAL rows exercise inference; TEST is never forwarded here.
    data = cfg["data"]
    ids = source_smoke_ids(groups, members, "val", 4) if joint else groups["val"][:4]
    ds = PocketDatasetV8(**{k:v for k,v in data.items() if k in inspect.signature(PocketDatasetV8).parameters},
                         pdb_ids=ids, use_nma=False, split="val", verbose=False)
    if ds.pdb_ids != ids:
        raise RuntimeError("Validation loader dropped samples")
    inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([ds[i] for i in range(len(ds))]), device)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs)
    if not torch.isfinite(outputs["logits"][inputs["protein_mask"].bool()]).all():
        raise RuntimeError("Non-finite FP16 validation output")
    if model._cached_classmass_hidden is not None:
        raise RuntimeError("Training graph retained after validation")
    clone = trainer.PocketGVPFusionV11m13(cfg["model"])
    clone.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()}, strict=True)
    verify_prepared(root)
    report.update(passed=True, train_pairs=batch["pdb_ids"], val_pairs=ids,
        metrics=metrics, scalars=writer.scalars, seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_training_allocated_gib=training_peak, test_forward_used=False,
        note="Smoke weights are discarded; the production job starts from fresh seed44 initialization")
    save_json(root/"smoke_pass.json", report)
    print("INTERNAL_GFGF_SMOKE_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

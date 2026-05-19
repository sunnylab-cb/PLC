"""
Multi-threaded feature range/stats + smart dim classification for normalization decision.

Coords are intentionally excluded:
  - ligand: loads only atom_features[109] (shape_3d dims 99-108 are derived descriptors,
            not raw xyz; coords/coords_heavy_current/coords_original are separate keys)
  - protein: loads physchem[62] and spatial_scalar[89] only (ca_coords is a separate key)
  - esm2:   loads embeddings[1280] only (atom37_positions is a separate key)
  - unimol: loads embedding[512] only (molecular-level, no coords)

Dim classification:
  BINARY            values only in {0, 1}  → skip norm
  SMALL_INT_CLASS   all-integer, ≤10 unique values, range ≤ 20  → skip norm
  NEAR_CONSTANT     std < 1e-3  → skip (useless)
  CONTINUOUS_OK     range ≤ 5 and std ≤ 2  → already reasonable scale, norm optional
  CONTINUOUS_NORM   range > 5 or std > 2  → normalize
"""

import argparse
import os
import glob
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


DATA_ROOT = "/hpc2hdd/home/shuang886/jhspoolers/PLC/data"

LABELS = {
    "BINARY":           "skip (binary 0/1)",
    "SMALL_INT_CLASS":  "skip (integer class)",
    "NEAR_CONSTANT":    "skip (near-constant)",
    "CONTINUOUS_OK":    "optional (range/std already small)",
    "CONTINUOUS_NORM":  "NORMALIZE",
}


# ── per-file loaders (coords excluded by design) ─────────────────────────────

def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def load_ligand(path):
    return _load(path)["atom_features"].numpy().astype(np.float32)   # [N, 109]

def load_esm2(path):
    return _load(path)["embeddings"].numpy().astype(np.float32)       # [L, 1280]

def load_unimol(path):
    return _load(path)["embedding"].unsqueeze(0).numpy().astype(np.float32)  # [1, 512]

def load_protein_physchem(path):
    return _load(path)["physchem_features"].numpy().astype(np.float32)       # [L, 62]

def load_protein_spatial(path):
    return _load(path)["spatial_scalar_features"].numpy().astype(np.float32) # [L, 89]


LOADERS = {
    "ligand":   (load_ligand,           109,  "ligand_features_v5",       "**/*_ligand.pt"),
    "esm2":     (load_esm2,             1280, "esm2_latents",             "**/*_protein.pt"),
    "unimol":   (load_unimol,           512,  "unimol_embeddings",        "**/*_unimol.pt"),
    "physchem": (load_protein_physchem, 62,   "complete_features_v4_nma", "**/*_conf0.pt"),
    "spatial":  (load_protein_spatial,  89,   "complete_features_v4_nma", "**/*_conf0.pt"),
}


# ── accumulator with batch Welford + unique-value tracking ───────────────────

class Accumulator:
    MAX_UNIQUE = 20   # if a dim has >20 distinct values → treat as continuous

    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.M2   = np.zeros(dim, np.float64)
        self.fmin = np.full(dim,  np.inf)
        self.fmax = np.full(dim, -np.inf)
        self.unique   = [set() for _ in range(dim)]   # None once overflowed
        self.overflow = np.zeros(dim, bool)
        self.all_int  = np.ones(dim, bool)

    def update(self, x: np.ndarray):
        N, D = x.shape
        if N == 0:
            return
        # batch Welford (parallel combine formula)
        bm = x.mean(axis=0)
        bM = ((x - bm) ** 2).sum(axis=0)
        if self.n == 0:
            self.mean[:] = bm
            self.M2[:]   = bM
        else:
            delta = bm - self.mean
            new_n = self.n + N
            self.mean += delta * (N / new_n)
            self.M2   += bM + delta**2 * (self.n * N / new_n)
        self.n += N
        np.minimum(self.fmin, x.min(axis=0), out=self.fmin)
        np.maximum(self.fmax, x.max(axis=0), out=self.fmax)
        # unique-value + integer tracking (loop only over D, not N×D)
        for d in range(D):
            col = x[:, d]
            if self.all_int[d] and not np.all(np.abs(col - np.round(col)) < 1e-4):
                self.all_int[d] = False
            if not self.overflow[d]:
                self.unique[d].update(np.round(col, 5).tolist())
                if len(self.unique[d]) > self.MAX_UNIQUE:
                    self.overflow[d] = True
                    self.unique[d] = None

    def finalize(self):
        std = np.sqrt(np.maximum(self.M2 / max(self.n - 1, 1), 0)).astype(np.float32)
        return {
            "mean":      self.mean.astype(np.float32),
            "std":       std,
            "min":       self.fmin.astype(np.float32),
            "max":       self.fmax.astype(np.float32),
            "n_rows":    self.n,
            "unique":    self.unique,    # list[set|None]
            "overflow":  self.overflow,  # bool array
            "all_int":   self.all_int,   # bool array
        }


# ── dim classifier ────────────────────────────────────────────────────────────

def classify_dim(d, s):
    mn, mx, std = float(s["min"][d]), float(s["max"][d]), float(s["std"][d])
    rng = mx - mn
    is_int   = bool(s["all_int"][d])
    overflow = bool(s["overflow"][d])
    unique   = s["unique"][d]  # set or None

    if std < 1e-3:
        return "NEAR_CONSTANT"

    if not overflow and unique is not None:
        clean = {round(v) if abs(v - round(v)) < 1e-4 else v for v in unique}
        if clean <= {0.0, 1.0}:
            return "BINARY"
        if is_int and len(unique) <= 10 and rng <= 20:
            return "SMALL_INT_CLASS"

    if rng > 5 or std > 2:
        return "CONTINUOUS_NORM"
    return "CONTINUOUS_OK"


# ── scan ──────────────────────────────────────────────────────────────────────

def worker(loader_fn, path):
    try:
        return loader_fn(path)
    except Exception:
        return None


def scan(paths, loader_fn, dim, n_workers, desc):
    acc = Accumulator(dim)
    errors = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(worker, loader_fn, p): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(paths), desc=desc, ncols=88):
            arr = fut.result()
            if arr is None:
                errors += 1
            else:
                acc.update(arr)
    s = acc.finalize()
    s["errors"] = errors
    return s


# ── report ────────────────────────────────────────────────────────────────────

def report(name, s, dim_names=None):
    D = len(s["mean"])
    counts = {k: 0 for k in LABELS}
    norm_dims, skip_dims, ok_dims = [], [], []

    for d in range(D):
        label = classify_dim(d, s)
        counts[label] += 1
        dname = dim_names[d] if dim_names else str(d)
        if label == "CONTINUOUS_NORM":
            norm_dims.append((d, dname, s["min"][d], s["max"][d], s["std"][d]))
        elif label == "CONTINUOUS_OK":
            ok_dims.append((d, dname, s["min"][d], s["max"][d], s["std"][d]))
        else:
            skip_dims.append((d, dname, label))

    print(f"\n{'='*68}")
    print(f"  {name}   n_rows={s['n_rows']}  errors={s['errors']}  dims={D}")
    print(f"{'='*68}")
    for k, v in counts.items():
        if v:
            print(f"  {k:<22} {v:>4} dims   {LABELS[k]}")

    if norm_dims:
        print(f"\n  ── dims needing normalization ({len(norm_dims)}) ──")
        for d, dn, mn, mx, std in norm_dims:
            print(f"    [{d:3d}] {dn:<35}  min={mn:9.3f}  max={mx:9.3f}  std={std:.3f}")

    if ok_dims:
        print(f"\n  ── continuous but already small-scale ({len(ok_dims)}) ──")
        for d, dn, mn, mx, std in ok_dims:
            print(f"    [{d:3d}] {dn:<35}  min={mn:7.3f}  max={mx:7.3f}  std={std:.3f}")

    if skip_dims:
        print(f"\n  ── skipped dims ({len(skip_dims)}) ──")
        for d, dn, label in skip_dims:
            uniq = s["unique"][d]
            uniq_str = str(sorted(uniq)[:8]) if uniq is not None else ">20 unique"
            print(f"    [{d:3d}] {dn:<35}  {label:<18}  unique={uniq_str}")

    return counts


# ── known dim names ───────────────────────────────────────────────────────────

LIGAND_109_NAMES = (
    [f"atom_type_{i}"       for i in range(11)] +   # 0-10
    [f"hybridization_{i}"   for i in range(6)]  +   # 11-16
    ["aromaticity"]                              +   # 17
    [f"ring_{i}"            for i in range(7)]  +   # 18-24
    [f"degree_{i}"          for i in range(5)]  +   # 25-29
    [f"formal_charge_{i}"   for i in range(3)]  +   # 30-32
    ["gasteiger_charge"]                         +   # 33
    ["crippen_logp"]                             +   # 34
    ["tpsa_contrib"]                             +   # 35
    ["rmsf"]                                     +   # 36
    ["flexibility_flag"]                         +   # 37
    ["atom_sasa"]                                +   # 38
    ["lipophilic_sasa"]                          +   # 39
    ["hbond_donor"]                              +   # 40
    ["hbond_acceptor"]                           +   # 41
    [f"chirality_{i}"       for i in range(2)]  +   # 42-43
    [f"rot_bond_ctx_{i}"    for i in range(3)]  +   # 44-46
    [f"metal_binding_{i}"   for i in range(4)]  +   # 47-50
    [f"stereochemistry_{i}" for i in range(3)]  +   # 51-53
    [f"morgan_{i}"          for i in range(43)] +   # 54-96
    ["mol_logp"]                                 +   # 97
    ["mol_tpsa"]                                 +   # 98
    [f"shape_3d_{i}"        for i in range(10)]     # 99-108
)
assert len(LIGAND_109_NAMES) == 109


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--max",       type=int, default=None,
                        help="Cap files per feature set")
    parser.add_argument("--output",    default="stats_output.pt")
    parser.add_argument("--features",  nargs="+",
                        default=["ligand", "esm2", "unimol", "physchem", "spatial"])
    args = parser.parse_args()

    results = {}

    for feat in args.features:
        loader_fn, dim, subdir, pattern = LOADERS[feat]
        paths = sorted(glob.glob(os.path.join(args.data_root, subdir, pattern), recursive=True))
        if args.max:
            paths = paths[: args.max]
        print(f"\n[{feat}] {len(paths)} files  dim={dim}")
        s = scan(paths, loader_fn, dim, args.workers, feat)
        results[feat] = s

        dim_names = LIGAND_109_NAMES if feat == "ligand" else None
        report(feat, s, dim_names)

    torch.save(results, args.output)
    print(f"\n\nFull stats saved → {args.output}")


if __name__ == "__main__":
    main()

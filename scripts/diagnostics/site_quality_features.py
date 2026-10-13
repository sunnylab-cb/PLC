"""Label-free candidate inputs; supervision is computed in a separate function."""

import numpy as np

from scripts.diagnostics.strict_val_hybrid_candidates import build_pools


STAT_NAMES = (
    "log_size", "protein_fraction", "observed_fraction", "mean_p", "std_p", "max_p",
    "q25_p", "median_p", "q75_p", "top20_p", "soft_f1_proxy", "predicted_fraction",
    "original_iou", "gyration_over20", "radius_over40", "log_protein_size",
    "original_protein_fraction", "protein_mean_p", "is_original",
)


def candidate_inputs(ca, probability, active, observed):
    """Keep the previous hybrid pool unchanged, including original mask at index 0."""
    ca = np.asarray(ca, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    active, observed = np.asarray(active, bool), np.asarray(observed, bool)
    n = len(probability)
    if n == 0 or ca.shape != (n, 3) or active.shape != (n,) or observed.shape != (n,):
        raise ValueError("Malformed candidate input shapes")
    if not np.isfinite(ca).all() or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Nonfinite coordinates or invalid probabilities")
    candidates = build_pools(ca, probability, active)["hybrid_union"]
    return candidates, candidate_stats(ca, probability, active, observed, candidates)


def candidate_stats(ca, probability, active, observed, candidates):
    """Score a fixed mask list, also usable for label-free refined masks."""
    ca, probability = np.asarray(ca, np.float64), np.asarray(probability, np.float64)
    active, observed = np.asarray(active, bool), np.asarray(observed, bool)
    n = len(probability)
    rows = []
    for i, indices in enumerate(candidates):
        p = probability[indices]
        size = len(indices)
        if size:
            radial = np.linalg.norm(ca[indices] - ca[indices].mean(0), axis=1)
            mean, std, maximum = p.mean(), p.std(), p.max()
            quantiles = np.quantile(p, [.25, .5, .75]).tolist()
            top = np.sort(p)[-max(1, int(np.ceil(.2 * size))):].mean()
            rg, radius = np.sqrt(np.mean(radial ** 2)) / 20., radial.max() / 40.
        else:
            mean = std = maximum = top = rg = radius = 0.
            quantiles = [0., 0., 0.]
        overlap = int(active[indices].sum())
        rows.append([
            np.log1p(size), size / n, observed[indices].sum() / max(1, size),
            mean, std, maximum, *quantiles, top,
            2 * p.sum() / max(1e-12, size + probability.sum()),
            overlap / max(1, size), overlap / max(1, size + active.sum() - overlap),
            rg, radius, np.log1p(n), active.mean(), probability.mean(), float(i == 0),
        ])
    stats = np.asarray(rows, dtype=np.float32)
    assert stats.shape == (len(candidates), len(STAT_NAMES)) and np.isfinite(stats).all()
    return stats


def candidate_targets(candidates, target):
    """Observed contact-label quality, not a claim that other sites cannot bind."""
    target = np.asarray(target, bool)
    sizes = np.array([len(c) for c in candidates], dtype=np.int32)
    tp = np.array([target[c].sum() for c in candidates], dtype=np.int32)
    denom = sizes + target.sum()
    f1 = np.divide(2. * tp, denom, out=np.zeros(len(tp), dtype=np.float32), where=denom > 0)
    return f1, tp, sizes


def pack_candidates(candidates):
    offsets = np.concatenate(([0], np.cumsum([len(c) for c in candidates]))).astype(np.int64)
    return np.concatenate(candidates).astype(np.int32), offsets

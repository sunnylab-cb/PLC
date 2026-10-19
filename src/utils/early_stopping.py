"""Validation stopping state, independent of the best-loss checkpoint selector."""

import math


class ValidationEarlyStopping:
    def __init__(self, metric="f1_c0"):
        if metric not in ("f1_c0", "loss"):
            raise ValueError("early_stopping_metric must be f1_c0 or loss")
        self.metric = metric
        self.best = math.inf if metric == "loss" else -math.inf
        self.bad_epochs = 0

    def update(self, metrics):
        value = float(metrics[self.metric])
        improved = math.isfinite(value) and (
            value < self.best if self.metric == "loss" else value > self.best
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def state_dict(self):
        return dict(metric=self.metric, best=self.best, bad_epochs=self.bad_epochs)

    def restore(self, checkpoint):
        """Restore matching state; never reuse a legacy loss-based patience for F1."""
        saved = checkpoint.get("early_stopping_state")
        if saved is not None and saved["metric"] == self.metric:
            self.best = float(saved["best"])
            self.bad_epochs = int(saved["bad_epochs"])
            if math.isnan(self.best) or self.bad_epochs < 0:
                raise ValueError("Invalid early-stopping checkpoint state")
            return True

        # Old checkpoints can provide a best value, but not the new monitor's
        # elapsed patience. Include the current metric: old latest files saved
        # best-metric metadata before updating it for the completed epoch.
        key = "best_val_loss" if self.metric == "loss" else "best_val_f1_c0"
        values = [checkpoint.get(key), checkpoint.get("val_metrics", {}).get(self.metric)]
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if finite:
            self.best = min(finite) if self.metric == "loss" else max(finite)
        else:
            self.best = math.inf if self.metric == "loss" else -math.inf
        self.bad_epochs = 0
        return False

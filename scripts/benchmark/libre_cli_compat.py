#!/usr/bin/env python3
"""Run the unmodified LiBRe CLI with explicit string labels in its input CSV.

Some pandas versions infer binary label strings as integers, losing leading
zeros and breaking LiBReDataset. This shim only controls CSV parsing; it does
not change labels, model parameters, feature tensors, thresholds or metrics.
"""

from pathlib import Path
import runpy
import sys


def main():
    import pandas as pd

    entry = Path(sys.argv[1]).resolve()
    sys.argv = [str(entry)] + sys.argv[2:]
    csv_path = Path(sys.argv[sys.argv.index("--test_csv") + 1]).resolve()
    original = pd.read_csv

    def read_csv_with_string_labels(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == csv_path:
            dtype = dict(kwargs.pop("dtype", None) or {})
            dtype["label_sequence"] = str
            kwargs["dtype"] = dtype
        return original(path, *args, **kwargs)

    pd.read_csv = read_csv_with_string_labels
    sys.path.insert(0, str(entry.parent))
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        pd.read_csv = original


if __name__ == "__main__":
    main()

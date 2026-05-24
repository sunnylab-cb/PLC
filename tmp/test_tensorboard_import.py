#!/usr/bin/env python3
"""
Minimal TensorBoard import/write diagnostic.

Run:
    conda run -n AIAA python -u tmp/test_tensorboard_import.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


def step(name: str, fn):
    print(f"\n[STEP] {name}", flush=True)
    try:
        result = fn()
    except BaseException as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"[OK] {name}", flush=True)
    return result


def main() -> None:
    print(f"python={sys.executable}", flush=True)
    print(f"version={sys.version}", flush=True)
    print(f"pid={os.getpid()}", flush=True)

    step("import torch", lambda: importlib.import_module("torch"))
    torch = sys.modules["torch"]
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)

    tb = step("import tensorboard", lambda: importlib.import_module("tensorboard"))
    print(f"tensorboard={getattr(tb, '__version__', 'unknown')}", flush=True)

    step(
        "import torch.utils.tensorboard",
        lambda: importlib.import_module("torch.utils.tensorboard"),
    )
    SummaryWriter = step(
        "from torch.utils.tensorboard import SummaryWriter",
        lambda: getattr(importlib.import_module("torch.utils.tensorboard"), "SummaryWriter"),
    )

    out_dir = Path(tempfile.mkdtemp(prefix="tb_import_test_", dir="/tmp"))
    print(f"log_dir={out_dir}", flush=True)

    writer = step("SummaryWriter(log_dir)", lambda: SummaryWriter(str(out_dir)))
    step("writer.add_scalar", lambda: writer.add_scalar("test/scalar", 1.0, 0))
    step("writer.flush", lambda: writer.flush())
    step("writer.close", lambda: writer.close())

    files = sorted(p.name for p in out_dir.iterdir())
    print(f"created_files={files}", flush=True)
    print("\nALL_OK", flush=True)


if __name__ == "__main__":
    main()

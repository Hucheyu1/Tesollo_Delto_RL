# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""检查 DG5F 视触觉数据集尺寸、触觉活跃率和元数据一致性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("datasets/dg5f_vtdex_pretrain")
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    total = 0
    active = np.zeros(20, dtype=np.float64)
    force_sum = np.zeros(20, dtype=np.float64)
    force_max = np.zeros(20, dtype=np.float64)
    for split in ("train", "val"):
        split_total = 0
        files = sorted((root / split).glob("*.h5"))
        print(f"[{split}] shards={len(files)}")
        for path in files:
            with h5py.File(path, "r") as h5:
                metadata = json.loads(str(h5.attrs["metadata_json"]))
                tactile = h5["tactile_binary"][:]
                forces = h5["tactile_force_norms"][:]
                if tactile.shape[1:] != (20,) or h5["rgb"].shape[1:] != (224, 224, 3):
                    raise ValueError(f"数据尺寸错误: {path}")
                n = len(tactile)
                split_total += n
                total += n
                active += tactile.sum(axis=0)
                force_sum += forces.sum(axis=0)
                force_max = np.maximum(force_max, forces.max(axis=0))
                print(
                    f"  {path.name}: samples={n}, "
                    f"active={float(tactile.mean()):.4f}, "
                    f"checkpoint={metadata.get('policy_checkpoint')}"
                )
        print(f"[{split}] samples={split_total}")

    if total == 0:
        raise RuntimeError(f"数据集为空: {root}")
    print(f"[all] samples={total}")
    print("[all] per-channel active ratio:")
    print(np.array2string(active / total, precision=5, separator=", "))
    print("[all] per-channel mean force (N):")
    print(np.array2string(force_sum / total, precision=5, separator=", "))
    print("[all] per-channel max force (N):")
    print(np.array2string(force_max, precision=5, separator=", "))


if __name__ == "__main__":
    main()

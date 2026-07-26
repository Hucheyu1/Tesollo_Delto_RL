# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""DG5F VTDex HDF5 数据集加载与协议校验。"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DG5FVTDexH5Dataset(Dataset):
    """按 shard 延迟打开 HDF5，避免把全部 RGB 数据载入内存。"""

    def __init__(
        self,
        files: list[Path],
        *,
        max_samples: int | None = None,
    ) -> None:
        if not files:
            raise FileNotFoundError("没有找到 DG5F VTDex HDF5 shard")
        self.files = [Path(path).resolve() for path in files]
        self.lengths: list[int] = []
        self.metadata: list[dict] = []
        for path in self.files:
            with h5py.File(path, "r") as h5:
                if h5.attrs.get("format") != "dg5f-vtdex-pretrain-v1":
                    raise ValueError(f"数据格式不受支持: {path}")
                if "rgb" not in h5 or "tactile_binary" not in h5:
                    raise ValueError(f"缺少 rgb/tactile_binary: {path}")
                if h5["rgb"].shape[1:] != (224, 224, 3):
                    raise ValueError(f"RGB 尺寸必须为 [N,224,224,3]: {path}")
                if h5["tactile_binary"].shape[1:] != (20,):
                    raise ValueError(f"触觉尺寸必须为 [N,20]: {path}")
                if len(h5["rgb"]) != len(h5["tactile_binary"]):
                    raise ValueError(f"RGB 与触觉样本数不一致: {path}")
                self.lengths.append(len(h5["rgb"]))
                self.metadata.append(
                    json.loads(str(h5.attrs.get("metadata_json", "{}")))
                )

        self.cumulative: list[int] = []
        running = 0
        for length in self.lengths:
            running += length
            self.cumulative.append(running)
        self.total_samples = running
        self.limit = (
            self.total_samples
            if max_samples is None
            else min(self.total_samples, int(max_samples))
        )
        if self.limit <= 0:
            raise ValueError("数据集为空")
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.limit
        if index < 0 or index >= self.limit:
            raise IndexError(index)
        file_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if file_index == 0 else self.cumulative[file_index - 1]
        local_index = index - previous
        handle = self._handles.get(file_index)
        if handle is None:
            handle = h5py.File(self.files[file_index], "r")
            self._handles[file_index] = handle

        image = torch.from_numpy(handle["rgb"][local_index]).permute(2, 0, 1)
        image = image.to(dtype=torch.float32).div_(255.0)
        image = (image - IMAGE_MEAN) / IMAGE_STD
        tactile = torch.from_numpy(handle["tactile_binary"][local_index]).to(
            dtype=torch.float32
        )
        return image, tactile

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


def discover_split(dataset_root: Path, split: str) -> list[Path]:
    return sorted((dataset_root / split).glob("*.h5"))

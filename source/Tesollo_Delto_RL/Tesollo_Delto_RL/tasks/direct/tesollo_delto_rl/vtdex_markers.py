# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""VTDex 采集与下游番茄任务共享的视觉方向标记。"""

from __future__ import annotations

import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils


def spawn_tomato_orientation_markers(
    *,
    object_asset,
    num_envs: int,
    marker_offsets,
    marker_radius: float,
    diffuse_colors,
    emissive_colors,
) -> None:
    """把两个非共线彩色点附着到每个番茄的可视表面。"""

    marker_offsets = tuple(tuple(float(value) for value in offset) for offset in marker_offsets)
    diffuse_colors = tuple(tuple(float(value) for value in color) for color in diffuse_colors)
    emissive_colors = tuple(tuple(float(value) for value in color) for color in emissive_colors)
    if not (
        len(marker_offsets) == len(diffuse_colors) == len(emissive_colors) == 2
        and all(len(values) == 3 for values in marker_offsets + diffuse_colors + emissive_colors)
    ):
        raise ValueError("番茄方向标记需要两组 3-D 偏移、漫反射颜色和自发光颜色")
    marker_radius = float(marker_radius)
    if marker_radius <= 0.0:
        raise ValueError("番茄方向标记半径必须为正数")

    marker_directions = torch.tensor(marker_offsets, dtype=torch.float64)
    if not torch.isfinite(marker_directions).all() or (
        torch.linalg.vector_norm(marker_directions, dim=-1) <= 1.0e-8
    ).any():
        raise ValueError("番茄方向标记偏移必须是有限、非零向量")

    marker_cfgs = tuple(
        sim_utils.SphereCfg(
            radius=marker_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                emissive_color=emissive_color,
            ),
        )
        for diffuse_color, emissive_color in zip(
            diffuse_colors, emissive_colors, strict=True
        )
    )

    rigid_prim_paths = tuple(str(path) for path in object_asset.root_physx_view.prim_paths)
    if len(rigid_prim_paths) != num_envs:
        raise RuntimeError(
            f"每个环境应有一个番茄刚体，实际得到 {len(rigid_prim_paths)}/{num_envs}"
        )

    stage = sim_utils.get_current_stage()
    rigid_prim = stage.GetPrimAtPath(rigid_prim_paths[0])
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    local_range = bbox_cache.ComputeLocalBound(rigid_prim).ComputeAlignedRange()
    if local_range.IsEmpty():
        raise RuntimeError(f"无法计算番茄可视边界: {rigid_prim_paths[0]}")

    bounds_min = torch.tensor(tuple(local_range.GetMin()), dtype=torch.float64)
    bounds_max = torch.tensor(tuple(local_range.GetMax()), dtype=torch.float64)
    visual_center = 0.5 * (bounds_min + bounds_max)
    visual_radii = 0.5 * (bounds_max - bounds_min)
    if not torch.isfinite(visual_radii).all() or (visual_radii <= 1.0e-5).any():
        raise RuntimeError(
            f"番茄可视边界无效: min={bounds_min.tolist()}, max={bounds_max.tolist()}"
        )

    surface_scale = torch.rsqrt(
        torch.sum(torch.square(marker_directions / visual_radii), dim=-1)
    )
    marker_positions = visual_center + marker_directions * surface_scale.unsqueeze(-1)
    print(
        "[INFO] 番茄方向标记位置: "
        f"center={visual_center.tolist()}, radii={visual_radii.tolist()}, "
        f"positions={marker_positions.tolist()}"
    )

    for rigid_prim_path in rigid_prim_paths:
        for marker_index, marker_cfg in enumerate(marker_cfgs):
            marker_prim_path = f"{rigid_prim_path}/VTDexOrientationMarker{marker_index}"
            marker_prim = stage.GetPrimAtPath(marker_prim_path)
            if not marker_prim.IsValid():
                marker_prim = marker_cfg.func(
                    marker_prim_path,
                    marker_cfg,
                    translation=tuple(
                        float(value) for value in marker_positions[marker_index]
                    ),
                )
            if not marker_prim.IsValid():
                raise RuntimeError(f"创建番茄方向标记失败: {marker_prim_path}")

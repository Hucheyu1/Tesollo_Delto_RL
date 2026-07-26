# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""DG5F 视触觉数据采集与 VTDex 下游任务共享的传感器约定。

这里的顺序属于数据集协议。采集、预训练和下游策略必须使用同一顺序，
因此不要在单个脚本中另行维护一份容易漂移的列表。
"""

from __future__ import annotations


DG5F_VTDEX_TACTILE_BODY_NAMES = (
    "rl_dg_5_4",
    "rl_dg_4_4",
    "rl_dg_3_4",
    "rl_dg_2_4",
    "rl_dg_1_4",
    "rl_dg_5_3",
    "rl_dg_4_3",
    "rl_dg_3_3",
    "rl_dg_2_3",
    "rl_dg_1_3",
    "rl_dg_5_2",
    "rl_dg_4_2",
    "rl_dg_3_2",
    "rl_dg_2_2",
    "rl_dg_1_2",
    "rl_dg_5_1",
    "rl_dg_4_1",
    "rl_dg_3_1",
    "rl_dg_2_1",
    "rl_dg_1_1",
)

DG5F_VTDEX_TACTILE_INDICES = tuple(range(20))
DG5F_VTDEX_CONTACT_THRESHOLD_N = 0.01

# 与 Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0 完全相同的侧视相机。
DG5F_VTDEX_CAMERA_EYE_LOCAL = (0.11, 0.36, 0.36)
DG5F_VTDEX_CAMERA_TARGET_LOCAL = (0.11, 0.00267, 0.36)
DG5F_VTDEX_CAMERA_RESOLUTION = (224, 224)
DG5F_VTDEX_CAMERA_FOCAL_LENGTH = 18.0
DG5F_VTDEX_CAMERA_FOCUS_DISTANCE = 0.45
DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE = 20.955
DG5F_VTDEX_CAMERA_CLIPPING_RANGE = (0.01, 1.5)

# 近球形番茄需要非共线的视觉方向标记，否则单目 RGB 无法辨认自转角。
DG5F_VTDEX_TOMATO_MARKER_OFFSETS = (
    (0.018, 0.028, 0.010),
    (-0.018, 0.028, -0.010),
)
DG5F_VTDEX_TOMATO_MARKER_RADIUS = 0.005
DG5F_VTDEX_TOMATO_MARKER_DIFFUSE_COLORS = (
    (0.02, 0.25, 1.0),
    (1.0, 0.85, 0.02),
)
DG5F_VTDEX_TOMATO_MARKER_EMISSIVE_COLORS = (
    (0.0, 0.02, 0.15),
    (0.12, 0.08, 0.0),
)

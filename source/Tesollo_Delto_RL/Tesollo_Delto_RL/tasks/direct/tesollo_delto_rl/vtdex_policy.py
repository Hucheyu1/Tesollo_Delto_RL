# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL actor matching VTDexManip's state/CLS projection topology."""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState


class VTDexActorModel(MLPModel):
    """Project proprioception and frozen VTDex CLS features before fusion.

    The upstream encoder policies separately map proprioception and the frozen
    representation to 128-D features, concatenate them, and feed the result to
    the policy MLP. The representation is 384-D for VT-JointPretrain and 512-D
    for V-CLIP; DG5F proprioception is 40-D.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        proprioception_dim: int = 40,
        vtdex_embedding_dim: int = 384,
        projection_dim: int = 128,
        **kwargs,
    ) -> None:
        self.proprioception_dim = int(proprioception_dim)
        self.vtdex_embedding_dim = int(vtdex_embedding_dim)
        self.projection_dim = int(projection_dim)
        if min(self.proprioception_dim, self.vtdex_embedding_dim, self.projection_dim) <= 0:
            raise ValueError("VTDex actor dimensions must be positive")

        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        expected_dim = self.proprioception_dim + self.vtdex_embedding_dim
        if self.obs_dim != expected_dim:
            raise ValueError(
                f"VTDex actor expected {expected_dim} observations "
                f"({self.proprioception_dim}+{self.vtdex_embedding_dim}), got {self.obs_dim}"
            )

        self.state_encoder = torch.nn.Linear(self.proprioception_dim, self.projection_dim)
        self.vtdex_projector = torch.nn.Linear(self.vtdex_embedding_dim, self.projection_dim)

    def _get_latent_dim(self) -> int:
        return 2 * self.projection_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        observation = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        observation = self.obs_normalizer(observation)
        proprioception = observation[..., : self.proprioception_dim]
        vtdex_embedding = observation[..., self.proprioception_dim :]
        return torch.cat(
            (
                self.state_encoder(proprioception),
                self.vtdex_projector(vtdex_embedding),
            ),
            dim=-1,
        )

    def as_jit(self) -> torch.nn.Module:
        return _VTDexExportModel(self)

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        return _VTDexExportModel(self, verbose=verbose)


class _VTDexExportModel(torch.nn.Module):
    """Deterministic raw-observation wrapper for JIT and ONNX export."""

    is_recurrent: bool = False

    def __init__(self, model: VTDexActorModel, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.proprioception_dim = model.proprioception_dim
        self.input_size = model.obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.state_encoder = copy.deepcopy(model.state_encoder)
        self.vtdex_projector = copy.deepcopy(model.vtdex_projector)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = self.obs_normalizer(observation)
        latent = torch.cat(
            (
                self.state_encoder(observation[..., : self.proprioception_dim]),
                self.vtdex_projector(observation[..., self.proprioception_dim :]),
            ),
            dim=-1,
        )
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @torch.jit.export
    def reset(self) -> None:
        pass

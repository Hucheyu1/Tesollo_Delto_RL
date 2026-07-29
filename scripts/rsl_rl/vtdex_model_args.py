"""Shared CLI support for selecting a frozen VTDex representation model."""

from __future__ import annotations

import argparse


VTDEX_MODEL_CHOICES = ("joint", "vision")


def add_vtdex_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add the VTDex model selector used by both training and playback."""

    parser.add_argument(
        "--model",
        choices=VTDEX_MODEL_CHOICES,
        default="joint",
        help=(
            "Frozen VTDex encoder used by VTDex tasks: 'joint' keeps the default "
            "VT-JointPretrain RGB+touch model; 'vision' uses V-CLIP, the "
            "paper's strongest pretrained pure-vision baseline."
        ),
    )


def apply_vtdex_model_selection(env_cfg, agent_cfg, model_mode: str) -> None:
    """Apply a validated model mode and keep visual-only logs separate."""

    if not hasattr(env_cfg, "vtdex_model_mode"):
        if model_mode != "joint":
            raise ValueError("--model vision is only supported by VTDex tasks")
        return

    env_cfg.vtdex_model_mode = model_mode
    if model_mode == "vision":
        old_embedding_dim = int(env_cfg.vtdex_embedding_dim)
        new_embedding_dim = int(env_cfg.vtdex_vision_embedding_dim)
        env_cfg.observation_space = (
            int(env_cfg.observation_space) - old_embedding_dim + new_embedding_dim
        )
        env_cfg.vtdex_embedding_dim = new_embedding_dim
        actor_cfg = getattr(agent_cfg, "actor", None)
        if actor_cfg is not None and hasattr(actor_cfg, "vtdex_embedding_dim"):
            actor_cfg.vtdex_embedding_dim = new_embedding_dim

        suffix = "_vision"
        if not agent_cfg.experiment_name.endswith(suffix):
            agent_cfg.experiment_name += suffix

    model_id_attr = "vtdex_model_id" if model_mode == "joint" else "vtdex_vision_model_id"
    repo_root_attr = "vtdex_repo_root" if model_mode == "joint" else "vtdex_vision_repo_root"
    print(
        "[INFO] VTDex model selection: "
        f"mode={model_mode}, model_id={getattr(env_cfg, model_id_attr)}, "
        f"embedding_dim={env_cfg.vtdex_embedding_dim}, "
        f"policy_observation_dim={env_cfg.observation_space}, "
        f"root={getattr(env_cfg, repo_root_attr)}"
    )

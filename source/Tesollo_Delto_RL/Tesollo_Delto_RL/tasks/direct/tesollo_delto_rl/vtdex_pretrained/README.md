# Bundled VTDexManip components

This directory contains the minimal upstream files needed by the DG5F
`reorient_down-vt_all_cls` reproduction.

Source:

- Repository: `/root/gpufree-data/VTDexManip`
- Upstream task: `tv_tasks/tasks/reorient_down.py`
- Upstream task config: `config/task_env/reorient_down.yaml`
- Upstream PPO config: `config/algos/ppo/reorient_down.yaml`
- Upstream policy: `model/ppo/policy.py` (`ActorCriticVTEncoder`)
- Upstream encoder wrapper: `model/backbones/pre_model.py`
- Task/model selector: `utils/hydra_utils.py`
- Model ID: `vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210`
- Upstream license: MIT; see `LICENSE`

Runtime files under `model/` are copied without behavioral changes. The
checkpoint SHA-256 is:

```text
f4976958716236a95e5d07ba9d131945d3ce4d8df35973da3e8cd2d2b571dd9e
```

The files under `reference/` are retained only to make the Isaac Gym to Isaac
Lab port auditable. Runtime code imports only the files under `model/`.

`assets/reorient_up/` contains the ten object directories referenced by the
upstream `reorient_down.yaml`. Each directory retains the copied `coacd_1.urdf`
and OBJ files. Its `coacd_goal.urdf` is a mechanically derived, visual-only
version of the same URDF; this reproduces the upstream goal actor's separate
non-interacting collision group in Isaac Lab while preserving its exact mesh.

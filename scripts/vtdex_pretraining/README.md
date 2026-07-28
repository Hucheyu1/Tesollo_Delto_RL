# DG5F 视觉-触觉采集、预训练与 Tomato 迁移

本目录参考 `/root/gpufree-data/VTDexManipPretraining` 的 VT-JointPretrain
（`vt20t-reall-tmr05-bin-ft-cls`）实现，为 DG5F 增加一条独立流程：

```text
Direct-v0 已训练策略
  -> 224×224 RGB + 20 路 DG5F link 接触 HDF5
  -> VT-JointPretrain 掩码重建
  -> 384 维 CLS encoder
  -> VTDex-Tomato actor
```

原 `Direct-v0` 和官方 `ViTacReal-all-210` 模型都不会被覆盖。

## 1. 环境

```bash
source /root/gpufree-data/isaac_ws/IsaacLab/env_isaaclab/bin/activate
cd /root/gpufree-data/Tesollo_Delto_RL
```

## 2. 用 Direct-v0 策略采集

先采训练集：

```bash
python scripts/collect_vtdex_pretrain_dataset.py \
  --task Tesollo-Delto-DG5F-Direct-v0 \
  --checkpoint /绝对路径/to/model.pt \
  --num_envs 16 \
  --num_steps 5000 \
  --split train \
  --dataset_root datasets/dg5f_vtdex_pretrain \
  --headless
```

再使用不同 seed 单独采验证集：

```bash
python scripts/collect_vtdex_pretrain_dataset.py \
  --task Tesollo-Delto-DG5F-Direct-v0 \
  --checkpoint /绝对路径/to/model.pt \
  --num_envs 16 \
  --num_steps 500 \
  --split val \
  --seed 121 \
  --dataset_root datasets/dg5f_vtdex_pretrain \
  --headless
```

采集器会动态附加传感器，不改变策略输入：

- 策略输入：自动读取 checkpoint 首层并兼容历史 84 维、当前 94 维
  `Direct-v0` 策略；
- 相机：与 Tomato 下游任务相同，eye=`(0.11, 0.36, 0.36)`，
  target=`(0.11, 0.00267, 0.36)`，RGB 224×224；
- 触觉：20 个 DG5F link 的净接触力，阈值 0.01 N，顺序由
  `vtdex_data.py` 唯一定义；
- 调试坐标轴：强制关闭；
- 目标番茄 mesh：默认移出相机视野；
- 番茄表面两个小彩点：默认保留，因为下游 Tomato 任务也使用它们来消除
  近球形物体的旋转歧义。它们不是调试坐标轴；可用
  `--no_tomato_markers` 做消融。

每个 HDF5 还保存 action、reward、done、物体位姿和 20 路连续接触力，
仅用于审计；预训练 loader 只读取 `rgb` 和 `tactile_binary`。

数据量可能很大。未压缩 RGB 每个样本约 147 KiB，建议先用较小
`num_steps` 检查画面和触觉活跃率，再做长时间采集。

## 3. 数据检查

```bash
python scripts/vtdex_pretraining/inspect_dataset.py \
  --dataset_root datasets/dg5f_vtdex_pretrain
```

重点检查：

- 20 路触觉是否不是全 0；
- 是否有某些通道永久为 1（阈值过低或自碰撞）；
- train 和 val 是否都存在；
- manifest 中策略 checkpoint、相机和触觉顺序是否正确。

## 4. VT-JointPretrain

从随机初始化开始：

```bash
python scripts/vtdex_pretraining/train.py \
  --dataset_root datasets/dg5f_vtdex_pretrain \
  --output_root outputs/vtdex_pretraining/dg5f_from_scratch \
  --epochs 400 \
  --batch_size 16 \
  --effective_batch_size 64
```

也可以从当前官方 `ViTacReal-all-210` 权重继续训练，通常比小规模仿真数据
上完全从头训练更稳：

```bash
python scripts/vtdex_pretraining/train.py \
  --dataset_root datasets/dg5f_vtdex_pretrain \
  --output_root outputs/vtdex_pretraining/dg5f_finetune \
  --init_checkpoint \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/vtdex_pretrained/model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210.pt \
  --epochs 100 \
  --batch_size 16 \
  --effective_batch_size 64
```

脚本保留参考代码的核心设置：ViT-Small 12 层、384 维、6 heads、图像
mask ratio 0.75、触觉 mask ratio 0.5、20 路 binary tactile 和 CLS token。
输出目录会自动形成 `VTDexJointEncoder` 所需结构：

```text
<output_root>/model/vitac/model_and_config/
  ├── <model_id>.json
  └── <model_id>.pt
```

`<output_root>` 是模型产物根目录，不需要包含
`model/vitac/vtt_reall.py`。运行时会从本项目内置的
`vtdex_pretrained` 读取网络实现，只从 `<output_root>` 读取配置和权重。

## 5. 兼容性校验

训练完成后运行：

```bash
python scripts/vtdex_pretraining/validate_model.py \
  --dataset_root datasets/dg5f_vtdex_pretrain \
  --vtdex_repo_root outputs/vtdex_pretraining/dg5f_finetune \
  --model_id '<训练日志打印的 model_id>'
```

只有看到 `[OK]`、输出为 `[batch, 384]` 且没有 NaN/Inf 后，再启动下游 RL。

## 6. 在 Tomato 任务使用新模型

通过环境变量切换模型，不修改也不删除原官方模型：

```bash
export TESOLLO_VTDEX_REPO_ROOT=/root/gpufree-data/Tesollo_Delto_RL/outputs/vtdex_pretraining/dg5f_finetune
export TESOLLO_VTDEX_MODEL_ID='<训练日志打印的 model_id>'

python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0 \
  --num_envs 16 \
  --headless
```

注意：`TESOLLO_VTDEX_REPO_ROOT` 虽沿用历史名称，设置的应是包含
`model/vitac/model_and_config/<model_id>.json/.pt` 的模型产物根目录，
不是必须包含完整源码的 VTDexManip 仓库。

不设置这两个环境变量时，Tomato 任务仍加载原来的
`vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210`。

## 建议实验

至少比较三组：

1. 官方 `ViTacReal-all-210` 冻结 encoder；
2. 官方权重在 DG5F 数据上继续预训练后冻结；
3. DG5F 数据从头预训练后冻结。

比较相同 Tomato RL seed 下的成功率、收敛速度和触觉活跃分布。仅凭预训练
重建 loss 不能判断哪一个下游效果最好。

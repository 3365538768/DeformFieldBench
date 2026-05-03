# DeformFieldBench

## 数据与 Checkpoint

- Dataset: [Physical Field Material Parameter 5000 on Kaggle](https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000)
- Checkpoints: [HandsomeHusky/DeformFieldBench on Hugging Face](https://huggingface.co/HandsomeHusky/DeformFieldBench/tree/main)

下载 checkpoint:

```bash
bash pretrained/download_models.sh
```

训练和评估默认读取以下结构：

```text
auto_output/dataset_5000/train/<sample_id>/sample_pack.npz
auto_output/dataset_5000/train_test_split.cleaned.json
pretrained/logic_baseline_epoch_0320.pt
pretrained/supervised_baseline_last.pt
pretrained/logic_abl_no_boundary.pt
pretrained/logic_abl_no_field_aux.pt
pretrained/logic_abl_no_multistage.pt
pretrained/logic_plus_phys_01.pt
```

`sample_pack.npz` 是统一数据接口。每个样本包含三视角 RGB deformation video、force mask、projected flow field、stress field、object mask，以及论文最终使用的四个材料参数：Young's modulus `E`、Poisson's ratio `nu`、density `rho`、yield stress `sigma_y`。

## 环境配置

推荐使用 Python 3.10。内部训练环境为 `envs/train`，核心版本为 `Python 3.10.20`、`torch 2.8.0+metax3.5.3.9`、`numpy 1.26.4`、`opencv-python 4.9.0.80`。如果使用 CUDA GPU，可安装与本机 CUDA 兼容的 PyTorch；如果使用 MACA/Metax 环境，需要配置 cu-bridge。

```bash
pip install -r requirements.txt
source scripts/setup_env.sh
```

平台环境可直接使用：

```bash
source scripts/activate_train_env.sh
```

`requirements.txt` 覆盖仿真和数据读写依赖，包括 `taichi==1.5.0`、`warp_lang==0.10.1`、`scipy==1.15.2`、`Pillow`、`pymeshlab`、`PyMCubes` 等。VLM benchmark 另需：

```bash
pip install -r requirements-vlm.txt
export DASHSCOPE_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ARK_API_KEY=your_key
```

## Simulation 与 Config

生成单个与最终 dataset 格式一致的样本：

```bash
PLY_PATH=/path/to/object.ply \
SIM_TYPE=press \
CONFIG=configs/simulation/train_config_dataset_full.json \
OUTPUT_PATH=outputs/simulation_sample \
bash scripts/run_simulation_sample_pack.sh
```

`SIM_TYPE` 可取 `press`、`drop`、`shear`、`stretch`、`bend`。输出目录会写入 `sample_pack.npz`，并保留 `meta/` 中的 GT 参数、边界条件、外力信息和 run 参数。模型最终输出的全局材料参数只包含 `E`、`nu`、`rho`、`sigma_y`；动作、边界、外力和 gravity 是仿真/重放元信息，不作为最终参数输出。

最终 dataset 形式只需要关注这些配置字段：

- `seed`、`num_simulations`、`sim_types`: 控制批量采样数量、随机种子和动作类型集合。
- `output_root`、`dataset_name`、`dataset_split`: 控制数据集路径，命名数据集格式为 `auto_output/<dataset_name>/<dataset_split>/<sample_id>/`。
- `base_config_by_sim_type`: 每个动作对应的基础仿真模板，例如 `press` 对应 `simulation/config/press_cube_jelly.json`。
- `num_views`、`num_render_views`、`random_render_views`: 控制多视角数量和随机视角采样。
- `render_outputs_per_sim_second`、`num_render_timesteps`: 控制时间采样；前者大于 0 时优先使用，按仿真时长决定输出帧数。
- `render_export_max_side`、`render_export_scale`、`camera_distance_scale`: 控制导出分辨率和相机距离。
- `render_img`、`output_view_stress_gaussian`、`output_view_flow_gaussian`、`output_view_force_mask`、`output_view_object_mask`: 控制最终 sample pack 中需要的 RGB、stress、flow、force mask、object mask。
- `stress_gaussian_single_channel`、`force_mask_single_channel`: 使用单通道场图和 mask，和训练配置中的输入约定一致。
- `pack_sample_pack`、`pack_sample_pack_include_object_mask`、`sample_pack_name`: 必须开启，用于写出 `sample_pack.npz`。
- `output_bc_info`、`output_force_info`、`output_initial_force_mask_arrow`: 写入边界和外力元信息，parameter ambiguity replay 会复用这些信息重放仿真。

## Arch4 训练

Arch4 是监督参数回归基线，入口为 `my_model.train`，配置为 `configs/my_model/train_dataset_5000_param_only.json`。该配置读取 `auto_output/dataset_5000/train` 和 `train_test_split.cleaned.json`，输入为 3 视角、64 帧、224 分辨率；训练时关闭场监督，只回归论文最终材料参数 `E`、`nu`、`rho`、`sigma_y`。

```bash
NUM_GPUS=8 bash scripts/train_my_model_param_only.sh
```

如果需要训练带场监督的 Arch4，不要使用上面的 `train_my_model_param_only.sh`，因为该脚本会强制传入 `--disable_aux_losses`。建议复制一份配置，只打开 Arch4 的辅助场头和场损失：

```bash
cp configs/my_model/train_dataset_5000_param_only.json \
  configs/my_model/train_dataset_5000_with_fields.json
```

在新配置中设置：

```json
{
  "model": {
    "use_aux_field_heads": true,
    "dec_h": 112,
    "dec_w": 112
  },
  "train": {
    "disable_aux_losses": false,
    "lambda_stress": 1.0,
    "lambda_flow": 1.0,
    "lambda_force": 1.0,
    "checkpoint": {
      "save_dir": "output_checkpoints/my_model_dataset_5000_with_fields"
    }
  }
}
```

然后直接启动训练：

```bash
source scripts/setup_env.sh
NUM_GPUS=8
python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" -m my_model.train \
  --config configs/my_model/train_dataset_5000_with_fields.json
```

此时总损失为参数回归损失加上 stress、flow、force 三个辅助场重建损失；最终参数输出仍然只包含 `E`、`nu`、`rho`、`sigma_y`，场监督只作为训练辅助信号。

关键训练项：

- `train.epochs=400`、`batch_size=1`、`lr=3e-4`。
- checkpoint 默认保存到 `output_checkpoints/my_model_dataset_5000_param_only/last.pt`。

## Logic Model 训练与消融

最终模型入口为 `logic_model.train`，配置为 `configs/logic_model/final_logic_413_5000_new.json`。模型使用 `logic_v2_dino`，读取 `sample_pack.npz`，从三视角 RGB deformation video 中预测 dense physical fields，并回归论文最终材料参数 `E`、`nu`、`rho`、`sigma_y`。

```bash
NUM_GPUS=8 bash scripts/train_logic.sh
```

最终模型配置：

- `arch=logic_v2_dino`: 使用冻结的 DINOv2 frame encoder。
- `num_views=3`、`num_frames=64`、`img_size=224`: 与最终 dataset 输入一致。
- `temporal_adapter_type=transformer`: 对每个视角建模时序 deformation cue。
- `fusion_dim=512`、`fusion_heads=8`: 融合多视角表示。
- `field_head_mode=shared_temporal`: 使用共享 dense-field decoder。
- `field_use_patch_tokens=true`、`field_use_multiscale_spatial=true`: 使用 DINO patch tokens 和多尺度 RGB 条件生成局部场。
- `field_use_shared_task_phys=true`、`field_use_geometry_residual=true`、`field_sequential_stress=true`: 对应论文中的 physics bottleneck、几何残差和 sequential stress refinement。
- 最终参数头预测 `[log(1+E), nu, log(1+rho), log(1+sigma_y)]`，其中大范围物理量在 log space 训练。

关键训练项：

- `train.epochs=320`、`batch_size=1`、`lr=1e-5`、`use_amp=true`。
- `lambda_stress=1`、`lambda_flow=1`、`lambda_force=2` 控制辅助场监督。
- `stage_flow_force_epochs=100`、`stage_stress_epochs=80`、`stage_joint_epochs=100` 控制多阶段训练。
- `lambda_stress_edge=0.05`、`lambda_flow_edge=0.2` 控制边界增强损失。
- `use_phys_loss=false`、`lambda_phys=0.0` 为默认最终模型设置。

消融实验建议从最终配置复制一个新 JSON，只修改消融相关字段，其它数据、模型和训练参数保持一致：

```bash
cp configs/logic_model/final_logic_413_5000_new.json \
  configs/logic_model/logic_abl_no_field_aux.json
# 按下面规则编辑 configs/logic_model/logic_abl_no_field_aux.json
CONFIG=configs/logic_model/logic_abl_no_field_aux.json \
NUM_GPUS=8 \
bash scripts/train_logic_ablation.sh
```

消融规则：

- `logic_abl_no_field_aux`: 将 `lambda_stress/lambda_flow/lambda_force` 设为 `0`，去掉辅助场监督。
- `logic_abl_no_multistage`: 将所有 stage epoch 设为 `0`，去掉多阶段训练日程。
- `logic_abl_no_boundary`: 将 `lambda_stress_edge/lambda_flow_edge` 设为 `0`，去掉边界增强。
- `logic_plus_phys_0001/001/01`: 打开 `use_phys_loss=true`，并设置 `lambda_phys` 为 `0.001/0.01/0.1`；发布 checkpoint 中包含 `logic_plus_phys_01`。

## 统一 Eval 与指标

所有模型使用同一套评估入口 `scripts/eval_param_metrics.sh`。`MODEL=my_model` 调用 `eval_abalation.eval_my_model`，`MODEL=logic` 调用 `eval_abalation.eval_logic`。

```bash
MODEL=logic \
CHECKPOINT=pretrained/logic_baseline_epoch_0320.pt \
OUT_DIR=outputs/param_eval/logic \
bash scripts/eval_param_metrics.sh
```

```bash
MODEL=my_model \
WEIGHTS=pretrained/supervised_baseline_last.pt \
OUT_DIR=outputs/param_eval/my_model \
bash scripts/eval_param_metrics.sh
```

常用参数：

- `EVAL_SPLIT=test`: 评估 split。
- `NUM_SAMPLES=0`: 使用全部样本；大于 0 时抽样。
- `OUT_DIR`: 输出目录。

输出包括 `summary.json`、`score.json`、`param_metrics/`、`field_metrics/`。`param_metrics` 只统计论文最终材料参数 `E`、`nu`、`rho`、`sigma_y`，指标包含 `MAE`、`RMSE`、`MAPE`、`median AE`、`P90 AE`、`bias`、`Pearson/Spearman`、`R2`、calibration slope/intercept，以及 range-normalized composite error。场指标对应论文中的 stress、motion flow、contact/force region，包含 `MSE`、`SSIM`、region IoU/recall、flow coverage/velocity/EPE/motion quality、force main recall/centroid distance/area error、stress hotspot recall/weighted MAE/rank correlation。

## Parameter Ambiguity Replay

该评估先使用模型预测参数，再把预测参数写回仿真配置，调用 `modified_simulation.py` 重放样本，最后比较重放 RGB 与 GT RGB 的 `SSIM` 和 `PSNR`。它直接复用参数评估导出的 `param_metrics/sample_records.jsonl`。

```bash
DATASET_ROOT=auto_output/dataset_5000/train \
RECORDS=outputs/param_eval/logic/param_metrics/sample_records.jsonl \
OUT_DIR=outputs/param_ambiguity/logic \
REPLAY_NUM_SAMPLES=0 \
REPLAY_NUM_GPUS=1 \
bash scripts/eval_param_ambiguity.sh
```

输出：

```text
outputs/param_ambiguity/logic/replay_selection.json
outputs/param_ambiguity/logic/replay_metrics.jsonl
outputs/param_ambiguity/logic/replay_metrics.csv
outputs/param_ambiguity/logic/replay_summary.json
outputs/param_ambiguity/logic/replay_samples/<sample_id>/
```

`REPLAY_NUM_SAMPLES=0` 表示重放全部记录；`REPLAY_SAMPLE_MODE=random` 可随机抽样，`REPLAY_SEED` 控制随机性。

## 目录说明

```text
configs/          训练、仿真和评估配置
simulation/       sample_pack 仿真生成代码
my_model/         Arch4 监督参数回归基线
logic_model/      Logic Model 和训练代码
eval_abalation/   参数、场和 ambiguity 评估
vlm_benchmark/    VLM benchmark 入口
scripts/          对外复现实验脚本
pretrained/       checkpoint manifest 和下载脚本
outputs/          默认本地输出目录
```


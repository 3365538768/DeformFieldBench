# DeformFieldBench / FieldBridge

**仓库根目录即当前 FieldBridge 项目。** 模型、训练、数据、评估和网页控制统一使用这套代码，原论文旧实现已从当前分支移除，可从 Git 历史提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c` 查看。克隆后进入 `DeformFieldBench`，无需再进入 `rsi_real_test` 子目录；下文配置和脚本路径均相对仓库根目录。A800 的部署目录仍为下一段所示位置。

核心入口：[模型](src/fieldbridge/model.py)、[训练与早停](src/fieldbridge/train.py)、[验证](src/fieldbridge/evaluation.py)、[实验队列](scripts/optimization_manager.py)、[Web 监控](monitor/README.md)、[G0/G1 配置](configs/)。

目标：静态材料先验 + 动态物理场推断可用于指定仿真器的材料参数，以未见受力和真实交互验证。项目位于 A800 的 `/mnt/afs_engine/linrui/rsi_real_test`。

详细路线见 [docs/OPTIMIZATION_PLAN_ZH.md](docs/OPTIMIZATION_PLAN_ZH.md)，当前验收记录见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md)。运行时状态和日志保存在部署目录，不随 Git 发布。

根目录迁移与独立运行验证见 [迁移说明](docs/ROOT_MIGRATION_2026-11-07_ZH.md)：不含旧代码的导出目录已通过 14 项测试及 CPU 训练→检查点评估全流程。

## 已实现范围

- 独立模块化实现，不依赖原论文的 Python 模块或仿真代码目录。
- 原版 `sample_pack.npz` 读取、可变视角/帧数补齐与 mask、按物体隔离 split、仅训练集归一化、缺失/不适用 yield mask。
- 本地权重 DINOv2/DINOv3 适配器、跨时间和视角 attention、顺序 contact/flow→stress 解码、field feedback 与等容量控制、先验参数融合接口。
- BF16、梯度累积、单卡/torchrun DDP、无重复样本的分布式验证、分阶段选模、紧凑检查点和 epoch 边界恢复；所有样本均进入指标，不按方法剔除尾部。
- G0/G1 顺序执行队列、共享冻结 RGB 特征缓存、联合验证停滞早停、网页阶段反馈及单独/全局暂停继续。操作与实测证据见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md) 和 [观测台说明](monitor/README.md)。
- 单独的原始度量场模式；不将旧 RGB flow/stress 可视化误称为物理单位场。

这是一套新架构的研究起点，不是原论文权重兼容重现，也不是已经证明有效的模型。VoMP 推理、手机视频重建、可微重放、FEM/MPM 标定与复合材料生成尚未实现；已经提供先验和仿真结果接口，具体状态不得混淆。`tiny` 和少步小模型只用于软件检查，不可作为论文结果。

## 环境与最短运行

推荐 Python 3.12；项目支持 Python 3.10+。A800 已验证环境为 PyTorch 2.8.0+cu128 / torchvision 0.23.0。先按设备安装匹配的 PyTorch/torchvision，再安装本项目；CPU 软件检查不需要 CUDA、数据集、预训练权重或原仓库源码。

从全新克隆运行 CPU 最小例子：

```bash
git clone https://github.com/3365538768/DeformFieldBench.git
cd DeformFieldBench
python3 -m venv .venv
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pip install -e '.[dev]'
python scripts/make_fixture.py --output data/fixture
python -m fieldbridge.cli fit-stats data/fixture/manifest.jsonl data/fixture/stats.json
python -m fieldbridge.train --config configs/smoke_cpu.json
python -m fieldbridge.cli evaluate --config configs/smoke_cpu.json \
  --checkpoint runs/software_smoke_cpu/best_material.pt --split val \
  --output runs/software_smoke_cpu/eval_val
python -m pytest -q
```

Windows 使用 `python -m venv .venv` 和 `.venv\Scripts\Activate.ps1`，线程设置可用 `$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'`。重复训练请使用新的 output 路径或显式恢复，已有检查点不会被静默覆盖。该例子是软件验证，不是物理仿真或材质预测效果证明。

真实 GPU 训练须另行准备数据清单、训练集统计和官方权重，随后在根目录运行：

```bash
python -m fieldbridge.train --config configs/g1_D_field_bridge_seed17.json
# 多卡配置：
GPUS=8 bash scripts/train_a800.sh configs/smoke_ddp8.json
```

G1 配置使用共享特征缓存：首次运行前执行 `python scripts/prepare_feature_cache.py --config configs/g1_C_aux_no_feedback_seed17.json`。它需要既有 G1 数据清单；当前 A800 的 G0/G1 迁移队列还需要对应检查点与运行回执。历史队列快照位于 `docs/examples/OPTIMIZATION_PLAN_A800_2026-11-07.json`，不会作为全新克隆的活动任务。独立训练入口不要求这些历史回执。

真实数据：HF 的 dataset repo 和 model repo 同名，但内容不同。数据 ZIP 来自 `https://huggingface.co/datasets/HandsomeHusky/DeformFieldBench`；不带 `/datasets/` 的地址提供旧权重。

```bash
python scripts/import_deformfield.py --root data/real_subset \
  --output data/manifests/new_manifest.jsonl
python -m fieldbridge.cli audit data/manifests/new_manifest.jsonl
python -m fieldbridge.cli fit-stats data/manifests/new_manifest.jsonl data/manifests/new_manifest.stats.json
python scripts/generate_experiments.py --base configs/p0_dinov2_giant.json --output configs/experiments_p0
```

新 split 默认按物体固定 seed 分配 80/10/10 左右。已有发布协议时应使用 `--original-split data/metadata/train_test_split.cleaned.json`：逐一校验 4,914 个原始 ID，保留 494 条原测试样本，仅从原训练物体中划出 val；声明在原划分内却缺少 pack 时立即报错。数据清单路径和统计路径必须同步改入配置。

完整数据审计入口为 `scripts/audit_full_dataset.py`，核对官方 SHA256、所提取成员 CRC、五个完整数组的内部 CRC/形状/类型，以及标签和真实采样时间。`scripts/run_p0_pipeline.py` 是等待本次下载完成后执行审计和索引的一次性任务，不运行模型测试集评估。阶段目标见 [目标与阶段验收](docs/TARGET_AND_STAGE_GATES_ZH.md)。

## 主干权重

```bash
python scripts/prepare_backbone.py --family dinov2 --variant vitg14 --download-dinov2
python scripts/prepare_backbone.py --family dinov3
```

DINOv3 的官方权重需要按 Meta 的许可/访问流程取得，再放入配置指定的路径；项目不绕过访问要求，也不会缺权重时静默用随机初始化。DINOv2-S 在这里仅用于验证真实数据训练。正式对照保留 DINOv2-g，候选 DINOv3-L/300M 与 H+/840M，分别比较质量和成本，不声称模型更新必然带来收益。

## 公平实验

`C_aux_no_feedback` 对 `D_field_bridge` 是首要对照；`E_prior_direct` 对 `F_prior_field` 是静态先验融合对照。生成器提供 3 seeds × 6 variants。后两组必须提供独立静态模型的预测和来源信息，缺失时拒绝训练。禁止把 GT 参数作为先验填充。

评估示例：

```bash
python -m fieldbridge.cli evaluate --config CONFIG.json --checkpoint runs/RUN/best_material.pt \
  --split val --output runs/RUN/eval_val
python scripts/bootstrap_object_ci.py runs/C/eval_test/predictions.jsonl runs/D/eval_test/predictions.jsonl
```

`--split test` 留到协议、模型和超参冻结后使用。输出 standardized MAE 是 train-standardized ln1p/linear 指标，不是原论文 SPE；不把不同指标直接并排排名。预测方差是待校准估计，不能默认解释为可信后验。

## 静态先验与仿真接口

`scripts/join_static_priors.py` 接收独立模型 JSONL，要求 sample_id、parameters、model、checkpoint_sha256、input_sha256、training_data_protocol。VoMP 的几何、单位、部件聚合和缓存需另做适配。

`fieldbridge.simulation.EngineAdapter` 规定 SI 位置、时间、接触力、能量和引擎版本；目前未附任何“假仿真”实现。待标定通过后再连接真实 PhysGaussian/libuipc 后端。

## 代码来源

原项目参考：`https://github.com/3365538768/DeformFieldBench`，提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c`。本项目依据公开接口与论文重写训练模块，不直接复制上游训练实现。原项目顶层未检出许可证，第三方代码和权重分别保留其许可，暂不为它们重新声明授权。DINOv3 使用 DINOv3 License，不能简单称为 Apache/MIT。

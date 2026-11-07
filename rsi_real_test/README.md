# FieldBridge / RSI Real Test

**这里是 DeformFieldBench 当前新增的重构项目。** 仓库根目录中的旧论文训练代码继续保留；本轮模型、训练、数据、评估和网页控制统一在此目录。使用本项目时先 `cd rsi_real_test`；下文配置和脚本路径均相对此目录。A800 的部署目录见下一段。

核心入口：[模型](src/fieldbridge/model.py)、[训练与早停](src/fieldbridge/train.py)、[验证](src/fieldbridge/evaluation.py)、[实验队列](scripts/optimization_manager.py)、[Web 监控](monitor/README.md)、[G0/G1 配置](configs/)。

目标：静态材料先验 + 动态物理场推断可用于指定仿真器的材料参数，以未见受力和真实交互验证。项目位于 A800 的 `/mnt/afs_engine/linrui/rsi_real_test`。

详细路线见 [docs/OPTIMIZATION_PLAN_ZH.md](docs/OPTIMIZATION_PLAN_ZH.md)，当前验收记录见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md)。运行时状态和日志保存在部署目录，不随 Git 发布。

## 已实现范围

- 独立模块化实现；原论文仓库置于 `upstream/DeformFieldBench` 作为锁定提交的参考，不修改其中源码。
- 原版 `sample_pack.npz` 读取、可变视角/帧数补齐与 mask、按物体隔离 split、仅训练集归一化、缺失/不适用 yield mask。
- 本地权重 DINOv2/DINOv3 适配器、跨时间和视角 attention、顺序 contact/flow→stress 解码、field feedback 与等容量控制、先验参数融合接口。
- BF16、梯度累积、单卡/torchrun DDP、无重复样本的分布式验证、分阶段选模、紧凑检查点和 epoch 边界恢复；所有样本均进入指标，不按方法剔除尾部。
- G0/G1 顺序执行队列、共享冻结 RGB 特征缓存、联合验证停滞早停、网页阶段反馈及单独/全局暂停继续。操作与实测证据见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md) 和 [观测台说明](monitor/README.md)。
- 单独的原始度量场模式；不将旧 RGB flow/stress 可视化误称为物理单位场。

这是一套新架构的研究起点，不是原论文权重兼容重现，也不是已经证明有效的模型。VoMP 推理、手机视频重建、可微重放、FEM/MPM 标定与复合材料生成尚未实现；已经提供先验和仿真结果接口，具体状态不得混淆。`tiny` 和少步小模型只用于软件检查，不可作为论文结果。

## 环境与最短运行

当前隔离环境：Python 3.12 / PyTorch 2.8.0+cu128 / torchvision 0.23.0；原论文环境另行保留，不混装 Warp/GS 编译依赖。

```bash
cd /mnt/afs_engine/linrui/rsi_real_test
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
python scripts/make_fixture.py --output data/fixture
python -m fieldbridge.cli fit-stats data/fixture/manifest.jsonl data/fixture/stats.json
# 使用新的 output 路径，避免覆盖已有检查点
python -m fieldbridge.train --config configs/smoke.json
GPUS=8 bash scripts/train_a800.sh configs/smoke_ddp8.json
```

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

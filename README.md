# DeformFieldBench / FieldBridge

**仓库根目录即当前 FieldBridge 项目。** 模型、训练、数据、评估和网页控制统一使用这套代码，原论文旧实现已从当前分支移除，可从 Git 历史提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c` 查看。克隆后进入 `DeformFieldBench`，无需再进入 `rsi_real_test` 子目录；下文配置和脚本路径均相对仓库根目录。A800 的部署目录仍为下一段所示位置。

核心入口：[模型](src/fieldbridge/model.py)、[训练与早停](src/fieldbridge/train.py)、[验证](src/fieldbridge/evaluation.py)、[实验队列](scripts/optimization_manager.py)、[Web 监控](monitor/README.md)、[G0/G1 配置](configs/)。

Claude 接手资料：[codex_plan/README.md](codex_plan/README.md)，包含截至 2026-11-09 的目标、已核实结果、调研与证据快照、核心 rebuttal 顺序、V2 转向授权及无空闲机器期间的暂停约束。资料包为静态交接快照，不代表远程运行状态已重新核实。

目标：静态材料先验 + 动态物理场推断可用于指定仿真器的材料参数，以未见受力和真实交互验证。当前项目部署在 A800-2 的 `/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`。

2026-11-09 rebuttal 更新（用户截止日期 11 月 15 日）：[逐项 reviewer 回应与执行方案](docs/REBUTTAL_PRIORITY_PLAN_2026-11-09_ZH.md)、[阶段与交付清单](docs/REBUTTAL_EXECUTION_PLAN_2026-11-09.json)、[文献/官方实现索引](docs/REBUTTAL_SOURCES_2026-11-09.json)。优先核实原论文运行版本，补“保留 bottleneck 与场监督、仅切断反馈”的重训对照、原 MPM 物理校验、独立方法公平比较。已完成本地源码审计；公开默认配置关闭 consistency loss，但论文实际运行是否使用仍待检查点/日志确认。现有 M2 不能证明桥接优于匹配无桥控制；不把后续 Qwen/多引擎计划当作原稿结论的证据。A800-2 与监控继续暂停，本轮没有启动 GPU 实验。[可复跑审计脚本](scripts/audit_rebuttal_provenance.py)不加载模型、不连接服务器。

范围已按用户新指令收紧为核心 rebuttal，见[止损与 V2 转向规则](docs/REBUTTAL_CORE_AND_V2_DECISION_ZH.md)。严格证据不支持实际收益时，先立即报告，再开始已获授权的 V2 构建；单种子负结果或宽置信区间不包装成已证伪。已提供[逐 reviewer 回复工作稿](docs/REBUTTAL_RESPONSES_WORKING_DRAFT.md)，明确标记缺失证据，尚不可直接提交。用户确认当前没有空闲机器，继续暂停，仅优化方案；后续优先顺序、所需材料和 mask/场特征旁路风险见[最小算力执行协议](docs/REBUTTAL_COMPUTE_MINIMAL_PROTOCOL_ZH.md)。

2026-11-08 研究更新：[可行性判断、文献实验核对与方法创新方案](docs/RESEARCH_FEASIBILITY_AND_METHOD_2026-11-08_ZH.md)。单视频快速预测有效材料响应有条件可行，任意视频唯一恢复全部真实参数不能保证。UniPixie 已覆盖共享表征、多求解器概率输出及离线拟合蒸馏；建议下一步验证“消除加载/几何混淆的响应后验＋未见干预下的引擎标定”，按 P0–P5 设置停止门槛。此为研究方案，尚未启动这些新实验；[来源与版本索引](docs/RESEARCH_METHOD_SOURCES_2026-11-08.json)可追溯。

补充研究：[Qwen共享材质理解、few-shot引擎/真实域适配、物理中间量及真实拟合数据四方向规划](docs/FOUR_IDEAS_RESEARCH_AND_PLAN_2026-11-08_ZH.md)。新增近邻工作 Analytic Dynamics 已有物理状态教师到视频学生蒸馏，AdaptiGraph 已有少量交互物性适配。建议以可验证的少样本迁移为主问题，真实拟合标签为数据基础，Qwen与物理教师通过匹配消融决定是否保留。A800-2及监控仍按用户要求暂停，本轮没有部署、下载权重或启动新实验。

2026-11-08 新增不使用桥接和物理场的独立路线，已完成三组顺序训练与审计：纯视频 MAE **0.76642**，加首帧先验 **0.77290**，再加混合后验 **0.77985**。静态先验未显示收益；概率模型名义95%区间实际覆盖89.77%，尚未校准。这是512训练/128验证、单种子的诊断，不是最终测试或优于历史最强控制的证据。已提供 RGB 视频→材参 JSON 的 [推理命令与实现说明](docs/NF_IMPLEMENTATION_ZH.md)，Web 显示训练、教师夹具与推理核验状态。离线教师目前仅在已知加载的线性夹具通过，原视频教师蒸馏与跨引擎适配仍待验证。

2026-11-08：M2 可见位移与时空场桥五组对照已完成。辅助监督无桥 MAE **0.68281**，预测场桥 **0.68672**，纯 RGB **0.69952**；场桥确实被使用，但仍未超过最强无桥控制。新审计定位到整体运动占主导、512条取样缺少同条件材料重复、三视角与单视频部署不一致等限制。详见 [M2结果与五方面瓶颈分析](docs/M2_FEEDBACK_AND_BOTTLENECKS_ZH.md) 和 [整体优化路线](docs/NEXT_OPTIMIZATION_2026-11-08_ZH.md)。旧路径仅保留为历史记录，迁移已停止。

详细路线见 [docs/OPTIMIZATION_PLAN_ZH.md](docs/OPTIMIZATION_PLAN_ZH.md)，当前验收记录见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md)。运行时状态和日志保存在部署目录，不随 Git 发布。

根目录迁移与独立运行验证见 [迁移说明](docs/ROOT_MIGRATION_2026-11-07_ZH.md)：不含旧代码的导出目录已通过 14 项测试及 CPU 训练→检查点评估全流程。

## 已实现范围

- 独立模块化实现，不依赖原论文的 Python 模块或仿真代码目录。
- 原版 `sample_pack.npz` 读取、可变视角/帧数补齐与 mask、按物体隔离 split、仅训练集归一化、缺失/不适用 yield mask。
- 本地权重 DINOv2/DINOv3 适配器、跨时间和视角 attention、顺序 contact/flow→stress 解码、field feedback 与等容量控制、先验参数融合接口。
- BF16、梯度累积、单卡/torchrun DDP、无重复样本的分布式验证、分阶段选模、紧凑检查点和 epoch 边界恢复；所有样本均进入指标，不按方法剔除尾部。
- G0/G1 顺序执行队列、共享冻结 RGB 特征缓存、联合验证停滞早停、网页阶段反馈及单独/全局暂停继续。操作与实测证据见 [G0/G1 执行说明](docs/G0_G1_IMPLEMENTATION_AND_CONTROL_ZH.md) 和 [观测台说明](monitor/README.md)。
- 单独的原始度量场模式；不将旧 RGB flow/stress 可视化误称为物理单位场。

这是一套新架构的研究起点，不是原论文权重兼容重现，也不是已经证明有效的模型。已实测 libuipc 的小规模力脉冲与悬臂梁标定，发现并显式修正版本材料系数映射差异；复杂接触、复合材料生成、手机视频重建及完整在线拟合仍未完成。`tiny` 和少步小模型只用于软件检查，不可作为论文结果。

## 环境与最短运行

推荐 Python 3.12；项目支持 Python 3.10+。A800-2 已验证 PyTorch 2.8.0+cu126 / torchvision 0.23.0，环境快照见 [新机环境](docs/environment-lock-a8002.txt)。该机驱动535需将 `/usr/local/cuda-12.6/compat` 放到 `LD_LIBRARY_PATH` 前部；旧 A800 的 cu128 环境仅为历史记录。先按设备安装匹配的 PyTorch/torchvision，再安装本项目；CPU 软件检查不需要 CUDA、数据集、预训练权重或原仓库源码。

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

DINOv3 的官方权重需要按 Meta 的许可/访问流程取得，再放入配置指定的路径；项目不绕过访问要求，也不会缺权重时静默用随机初始化。已授权的 ViT-L/16 现可用 `pip install -e '.[dinov3]'` 和 `scripts/prepare_dinov3_hf.py` 下载固定 HF revision，校验 LFS SHA256、严格转换及双实现特征一致性；操作见 [G1f 执行说明](docs/G1F_DINOV3_EXECUTION_ZH.md)。DINOv2-S 用于前阶段诊断；更换骨干后会重训本骨干的 RGB 基线，不声称模型更新必然带来收益。

## 公平实验

最新按序执行与验收见 [D3 / I1 / M3 / B1 反馈](docs/D3_I1_M3_EXECUTION_ZH.md)。22组单视频控制训练及审计已完成，场桥尚未稳定胜过最强无场控制；受控小应变夹具核验通过，但不代替原MPM重放。BF16缓存候选在16条验证探针上保持点预测一致、体积减少约56%，尚未准入训练。网页显示真实阶段、结果与每任务暂停/继续，后续骨干适配、跨引擎头和在线公平比较仍待实施。

新增 [NF 严格零场路线](docs/NF_IMPLEMENTATION_ZH.md)：独立视频模型、首帧物体OOF静态先验、混合概率后验，以及带来源/准入约束的多解教师蒸馏接口。三组N0→N1→N2按相同预算顺序运行；既有小应变夹具已验证逆拟合教师到学生的流程，原GS视频教师尚未准入，不能将夹具结果当作真实视频蒸馏效果。

G1–G1e 已完成，预测场尚无独有收益。G1e 修复门值整体塌缩及常量捷径后，真场诊断 MAE 降至 0.67557；预测场 0.78649、等容量特征控制 0.77761。BCE+Dice 配对结果为 0.78008 / 0.77713，仍未通过验收。G1f DINOv3 匹配对照已完成：预测场 0.802526、等容量特征 0.802460、RGB 0.805450，未获得场桥独有收益。已转向 G2 双引擎 SI 响应矩阵与三种子适配控制，网页显示实际进度并支持暂停/继续，新增双引擎三维位移预览与实际求解轨迹播放。见 [G1f 反馈](docs/G1F_FEEDBACK_ZH.md) 和 [G2 协议](docs/G2_CROSS_ENGINE_PROTOCOL_ZH.md)。不自动重启旧全量训练。见 [G1f 执行说明](docs/G1F_DINOV3_EXECUTION_ZH.md)、[G1e 完成反馈与数据升级](docs/G1E_FEEDBACK_AND_DATA_UPGRADE_ZH.md)、[跨引擎 adapter 方案及实现边界](docs/CROSS_ENGINE_MATERIAL_ADAPTER_ZH.md)。

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

`fieldbridge.simulation.EngineAdapter` 规定 SI 位置、时间、接触力、能量和引擎版本。`scripts/calibrate_uipc.py`、`scripts/calibrate_uipc_beam.py` 运行实际 CUDA FEM，保留原生失败与显式系数修正后的校准结果；这尚不是完整数据生成后端。

`fieldbridge.engine_domains` 提供引擎/版本/本构/系数/数值协议/单位域和专属参数头原型。现有单头训练遇到多个原生标签域会拒绝混合；G2 已接入独立的受控场训练器：共享场编码器、训练集共享响应辅助、匹配对齐和引擎原生系数头。两个实际引擎的标定已通过，324 个案例和 9 组三种子训练已完成，180 次未见载荷回放和 20 次视频可见性求解已完成；尚未接入 RGB 视频主训练器。不同引擎的同名 GT 不默认等价。

## 代码来源

原项目参考：`https://github.com/3365538768/DeformFieldBench`，提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c`。本项目依据公开接口与论文重写训练模块，不直接复制上游训练实现。原项目顶层未检出许可证，第三方代码和权重分别保留其许可，暂不为它们重新声明授权。DINOv3 使用 DINOv3 License，不能简单称为 Apache/MIT。

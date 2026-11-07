# A800 项目搭建与分阶段优化方案

日期：2026-11-07。目标目录：`/mnt/afs_engine/linrui/rsi_real_test`。

## 1. 总目标和当前判断

面向真实短视频，输出指定仿真器中的全局/部件材料参数、不确定性，以及能预测新交互的仿真对象。视频可以是静态扫描或受挤形变；两者可辨识的信息量不同，必须在输出和实验中区分。

核心研究假设：**在相同输入、监督和训练预算下，显式 field feedback 比直接 RGB 回归或仅辅助场监督，更能纠正静态外观材料先验，并改善未见物体、未见受力与真实响应。**

最新目标进一步明确为：无在线拟合的模型在新动作响应上接近在线仿真持续拟合方法，并显著优于强无 bridge 对照。详细比较条件、开发目标和阶段出口见 `TARGET_AND_STAGE_GATES_ZH.md`。现阶段尚无达成该目标的质量证据。

先验证假设，再扩大模型和数据。静态先验、动态证据、模型规模、生成引擎四个因素分别做实验，不把一次联合改动的收益都归因于 field。

## 2. 已核实的项目与数据事实

- A800：8×NVIDIA A800-SXM4-80GB，驱动 570.148.08，CUDA SDK 12.8；检查时 GPU 空闲。AFS 约剩 1.6TiB，本地根分区约剩 193GiB。
- 参考代码：[DeformFieldBench](https://github.com/3365538768/DeformFieldBench)，锁定提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c`。
- 用户提供的同名 HF 地址有两个含义：model repo 是旧模型权重；[dataset repo](https://huggingface.co/datasets/HandsomeHusky/DeformFieldBench) 中为 `dataset.zip`，大小 41,425,509,436 字节。
- ZIP 中央目录实际列出 5,000 份 gt.json、4,914 份 sample_pack.npz；原 `train_test_split.cleaned.json` 明确只使用 4,914 条（4,420 train、494 test）。其余 86 份标签目录不在原训练/测试划分内，不能直接认定训练丢样。中央目录存在不代表剩余 NPZ 都已验证有效。
- 当前抽取了 7 个不同物体的小样本。它们仅用于格式和训练验收，不是随机抽样研究测试集；没有下载全部数据。
- 真实 NPZ 为 `[3,64,3,224,224]` uint8 的 RGB/stress/flow/force/object mask。即使 stress 源为灰度，也以三通道封装；旧版本监督不能直接视为 Pa 或 m/s。
- 原主干已是 `dinov2_vitg14_reg`。原配置 `dino_out_dim=1024` 是接口/投影设置，不宜直接当作主干真实 embedding 维度；新适配器从模型对象读取。
- 原发布配置含 `eval_split=train`，且默认 `use_phys_loss=false`。因此新训练必须拆分真正 val；不能声称当前主模型一定受旧弱物理损失影响，也不能仅凭该配置认定原论文全部实验的划分有误。
- 当前样本中的 jelly/plasticine yield 为零，不是这两个本构族都具有可观测零屈服应力的证据。新导入器只对 metal 启用 sigma_y 监督，后续 v2 数据按本构的真实定义重新明确参数表。
- 最新补充：Kaggle 第 2 版 108 份 PLY 已下载并覆盖发布 split 统计的 108 个物体，完成 CRC/哈希/数组检查。论文“100 个”与实际发布对象数的差异需要披露。两轮真实小样本 C/D 工程试跑已完成，但未显示参数收益；当前结果与下一步诊断见 `STAGE_P0_PROGRESS_ZH.md`。

## 3. 关键仿真诊断：需要优先处理耗散

检查用户发布代码和真实样本后，比之前仅查看上游更明确：

1. 示例 jelly 的 `rpic_damping=0.35`、`grid_v_damping_scale=0.999`；metal 为 0.25/0.9995；plasticine 为 0.45/0.9992。阻尼设置与材料类别绑定，可能成为本构分类的捷径，也可能让刚度与耗散补偿。
2. `simulation/mpm_solver_warp/mpm_solver_warp.py` 在每次子步中、网格归一化/重力更新后，对 `<1` 的 damping scale 施加速度缩放。
3. 已读取的真实样本 `000000/meta/run_parameters.json` 确认 `substep_dt=1e-4`、`frame_dt=.01`，不仅是模板默认。`gamma=-ln(d)/dt`，d=0.999 对应约 10.005/s，单独阻尼项的速度半衰期约 0.0693s；d=0.9995 在相同 dt 下对应约 5.001/s。这里计算的是孤立阻尼项，不是完整受重力/弹性/接触系统的实测衰减；其余样本由全量审计逐项核实。
4. 固定“每步乘 d”的方式会让改变 dt 同时改变单位物理时间的阻尼。应改为显式物理时间参数 `d(dt)=exp(-gamma*dt)`，或者使用经标定的粘性/黏弹本构，分别记录数值耗散与物理耗散。
5. 这为低回弹提供了具体候选原因，但尚未运行校准仿真，不能把它写成已经确认的唯一根因。

优先对同一标准物体、相同 E/nu/rho 做网格/时间步收敛与阻尼扫描：APIC/PIC 混合、gamma、接触边界、分辨率分别变化。记录自由落体轨迹、接触反力、回弹高度、能量和振动衰减。不要将多个设置同时改动后无法归因。

执行器的速度覆盖仍需改进：运动学驱动应记录反作用力；力控制与位移控制分开建模。准静态位移控制下，E 的整体缩放可能主要体现在反力，而非变形形状。若只监督视觉形状，不能期待所有绝对物理参数都可辨识。

另外，`set_velocity_on_cuboid` 的 `reset=1` 分支会在结束后的 15 个 dt 内对整个网格写零，分支中没有再次限制 cuboid 区域。这是启用时需要修正的代码风险；本次检查的五个标准模板均未启用它（reset=0 或不使用该执行器），所以不能把它归为这些模板当前低回弹的原因。

## 4. 模型升级：建议 DINOv3-L 起步，H+ 作为主候选

官方来源：[DINOv3](https://github.com/facebookresearch/dinov3)、[DINOv2](https://github.com/facebookresearch/dinov2)、[VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)。DINOv3 官方标称 L=300M、H+=840M、7B=6,716M。代码和权重采用 DINOv3 License，权重有官方申请访问流程；不能将其等同于无条件 MIT 开源。

| 档位 | 角色 | 训练方式 | 进入条件 |
|---|---|---|---|
| DINOv2-g（原主干） | 固定控制组 | 先冻结，训练新解码/融合模块 | 旧数据协议重建完成 |
| DINOv3-L/300M | 第一升级候选 | 冻结前端；再解冻最后 2–4 blocks | 已取得合法权重、验证输入与精度 |
| DINOv3-H+/840M | 更高容量候选 | 冻结或末层微调，BF16 | L 的消融有效且显存/吞吐 profile 通过 |
| DINOv3-7B | 备选教师 | 离线特征/蒸馏优先 | H+ 的收益仍明显，数据规模足够，且额外实现分片/离线缓存 |
| VideoMAEv2 | 时序路线对照 | 先独立 global 动态分支 | 前三项稳定后，避免初期加入第二套复杂大主干 |

原来已经是 giant，不应靠“名字更大”决定换模型。DINOv3-L 更小，但新预训练与局部特征可能更有效，是否更强必须在我们的参数和响应任务上确认。7B 不作为初始端到端训练目标：5k 级数据、稠密多帧和反复消融的成本更值得优先投入数据质量。

当前实现支持 DINOv2/DINOv3 本地仓库与权重、冻结主干、最后 blocks 解冻。LoRA、FSDP、teacher 蒸馏和 VideoMAEv2 还属于后续项，不能声称已具备。

### 4.1 架构

`RGB [B,V,T,3,H,W] → pretrained patch encoder → temporal/view masked attention → physics bottleneck`

随后两条路径：

- `contact + motion → conditioned stress → field encoder → parameter regressor`。
- 静态 VoMP/PIXIE 独立预测 → 带来源和 mask 的材料先验 → 与动态 embedding 融合、预测残差。

当前没有固定 view-id embedding，使用 masked attention，可处理不同视角数量并避免把复制单视图视作真正多视角。时间使用显式位置编码；新导入器优先从 run_parameters 的 render_frame_indices 和 frame_dt 恢复相对秒数，验证长度和单调性。只有缺少真实时间信息的旧包才使用归一化序列时间。相机外参/三维投影融合将在真实重建阶段增加。

初期输出 E、nu、rho、有效 sigma_y 及 3 类本构。正参数用 ln(1+x)，nu 线性，再按 train 统计标准化。输出均值和受限 log variance，先报告实际覆盖率，之后在独立 val 上校准；目前方差不能自动视为可信后验。

### 4.2 两种 field 数据模式

- `legacy_visual`：contact 1 + RGB flow 3 + RGB stress 3；只代表视觉响应监督。
- `metric`：contact 1 + vx/vy 2 + stress 1，保存浮点和独立 mask，明确 pixel/s、Pa 和尺度。flow 的训练缩放和 stress 的 log 缩放固定由训练集或物理常数确定。完整应力张量/3D velocity 后续另建 v2 schema。

同一实验禁止混用两种编码；不能将彩色光流反解后伪称原始速度，也不能对归一化 stress 图直接施加三维 PDE。

## 5. 重构边界与目录

```text
src/fieldbridge/
  config.py       严格配置与预训练权重路径校验
  data.py         NPZ/schema、缺失标签、视角/帧 mask、仅训练集统计
  model.py        backbone、时空融合、顺序场预测、反馈/先验分支
  losses.py       masked 参数/场损失、DDP 有效标签计数
  train.py        BF16、梯度累积、DDP、val 选模、检查点
  evaluation.py   不过滤样本指标、逐样本记录与来源分层
  simulation.py   未来引擎适配的 SI 结果规范，无伪后端
scripts/
  import_deformfield.py     原始数据导入与物体 split
  fetch_sample_subset.py   HTTP range 获取小样本、CRC 校验
  join_static_priors.py    独立静态预测与 provenance 合并
  generate_experiments.py  3 seeds × 6 受控实验
  bootstrap_object_ci.py   按物体配对 bootstrap
  prepare_backbone.py     官方主干、锁定提交和显式权重
  train_a800.sh           torchrun 入口
configs/                 原主干与 DINOv3 L/H+ 方案
tests/                   数据泄漏、GT 隔离、梯度路径、视角不变性
external/ weights/       官方依赖与权重，不混入自有训练模块
data/ runs/ docs/        清单/小样本、实验输出、方案与记录
```

新实现避免直接复用原 180KB 单体训练脚本，也不引入其 LMDB、GS CUDA、VLM API 依赖到基础训练环境。2026-11-07 已将 FieldBridge 提升为仓库根目录，旧论文实现从当前分支移除，历史版本仍可通过提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c` 获取。未重写或清除 Git 历史；A800 的项目部署路径保持 `/mnt/afs_engine/linrui/rsi_real_test`。

## 6. P0：数据审计与 field 独立贡献

### 6.1 第一批任务

1. 固定当前上游提交和数据 archive metadata/hash；列出 86 个缺 pack 的样本，不补零、不静默跳过。
2. 完整提取时只优先保留 NPZ、gt、元数据和 split；MP4 留作可选审阅，避免重复存储所有中间 PNG。
3. 检查全部可用样本 shape、NaN、mask 非空、字段时间/相机同步、物体来源和材参范围。明确 scale、dt、damping、gravity、contact 的实际值。
4. 对原论文测试物体使用明确列表，开发 val 从其 train 中按物体划出。随机生成的新 split 用于新实验，不能标成原论文复现。
5. 如需复现旧数值，运行锁定的旧模型和旧评估器，单独保存。新架构与新指标不与旧 SPE 混称同一实验。

### 6.2 主要控制组

| 实验 | field 监督 | 反馈给参数 | 静态先验 | 解释 |
|---|---|---|---|---|
| B RGB | 无 | 无 | 无 | 视频直接回归 |
| C Aux | 有 | 无 | 无 | 保留 bottleneck 和辅助任务 |
| D Bridge | 有 | 有 | 无 | D−C 识别 feedback 收益 |
| E Prior-direct | 有 | 无 | 有 | 静态先验 + 视频 |
| F Prior-field | 有 | 有 | 有 | F−E 识别先验条件下的 feedback 收益 |
| G Feature control | 有 | 等维特征代替场 | 无 | 排除额外分支容量解释 |

所有变体实例化相同模块和参数量，通过路由开关控制。C/D、E/F 保持训练阶段和随机种子完全一致。B 不做无意义的 field-only 预热，故 B 不作为唯一因果对照。还需对 contact/flow/stress 做逐项重训消融；其配置将在首批 C/D 实验确认后增加。

当前 generator 生成 18 个实验配置，但尚未运行完整科研训练。E/F 若没有独立先验将拒绝训练。已有推理置零只作为依赖性分析，不替代重训。

### 6.3 起始超参（待验证，非优化后的最终值）

- 输入 3 views × 16 frames × 224²；field 输出 112²。
- 冻结 backbone，AdamW heads lr=3e-4、weight decay=.05；解冻最后 blocks 时 backbone lr=3e-6。
- BF16、每卡 batch=1、8卡、accumulation=2，有效 batch=16；起始 field warmup=10 epochs + joint 至 80 epochs。
- field contact/flow/stress 权重初始各 1，参数 Huber=1，NLL=.05，本构 CE=.1。新旧指标/权重不同，不能据此宣称原论文完全重现。
- 先将 GPU 峰值控制在约 65–70GiB 以下，留出数据/通信余量。16→32→64 frames 和 224→448 逐项扩，不同时加大。
- 真正的 patch 特征缓存能节省冻结前端，但当前尚未实现；改动增广、尺度或 backbone 时必须换 cache key。
- 所有调参只使用 val，test 保持冻结。3 seeds 为 17/29/43。

验收：D 相对 C 在未过滤指标、尾部失败及至少一个未见动作上有稳定收益，且物体级配对区间支持该结论。若不成立，先修标签、激励和可辨识性；不直接靠更大网络掩盖。

## 7. P1：强主干与静态材料先验

先固定同一架构、分割和训练预算，比较 DINOv2-g 与 DINOv3-L/H+；必要时另列按 wall-clock 匹配的效率实验。不要让 backbone 升级同时改变时间层数、输入分辨率和样本量。

VoMP 优先、PIXIE 备选：

1. 先在它们原生资产上验证安装、权重和单位，再接我们的资产。检查公开 split、许可和资产重叠。
2. 对共享几何的“材料推断实验”，各模型获得相同三维几何和渲染视角；端到端视频重建另列，计入重建时间。
3. 输出部件/体素 E、nu、rho 及置信/有效 mask。第一阶段按已知区域聚合；不能直接把不同材料的 E 算术平均当等效整体刚度，应报告聚合假设或使用响应拟合。
4. 生成带 checkpoint/input hash 的 prior JSONL。若在本数据微调静态模型，只用 train；训练样本上的 prior 尽量用物体级 out-of-fold，防止动态模型依赖过拟合先验。
5. 做同外观不同材参、同材参不同纹理、先验故意错误三种子集。F 必须能在观察充分时纠正 E 的偏差，静态/微弱运动时保持合理不确定性。

当前已实现 prior 参数合同、融合和检查，**尚未安装/运行 VoMP 模型**。不能将“预留接口”写成已经完成 VoMP 集成。

## 8. P2：真实与未见动作验证

数据优先级：[PokeFlex](https://pokeflex-dataset.github.io/) → [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) → [EMPM](https://github.com/rai-opensource/empm) 的适用数据/方法。PokeFlex 有 wrench/contact 更便于标定，但不等于每个对象具有完整连续体真值。

协议：

- 推断只看一次 press/poke，测试 release、stretch、另一位置/幅度作用；未来帧与输入帧分开。
- 输入分 RGB-only、共享几何/RGB-D、标定受力三个轨道；不将额外深度/力信息免费提供给一方。
- 方法分前馈与固定预算 TTO。对各初始化使用同一优化器、目标和迭代预算；报告全部预处理与重建耗时。
- 与 spring-mass 方法比较时先比较共同动作下的几何/轨迹响应，不把 spring k 和 E 强行按数值比较。
- 保留纯物理重放作为材参检验；神经残差可以吸收视觉误差，但不能让其完全补偿错误材参后仍声称识别正确。
- 评价前景轨迹误差、Chamfer/表面距离、回弹/频率/衰减、接触与力—位移；PSNR/SSIM 为辅助。用按物体的配对 bootstrap，不能按帧伪增样本量。

可辨识性处理：已知质量和几何时优先固定 rho；无明显侧向应变时 nu 保留先验范围；未出现屈服时不输出假精确 sigma_y；未知力下报告参数-作用联合不确定性。

## 9. P3：经标定的复合材料数据

先保留校准 MPM，再用 [libuipc](https://github.com/spiriMirror/libuipc) 的 FEM/壳/接触能力建立高质量小集。[Newton](https://github.com/newton-physics/newton) 和 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 为吞吐候选，须在相同测试套件中比较。

标定套件：单轴拉压、悬臂梁、自由振动、标准跌落、质量体积能量漂移、dt/grid 收敛、接触穿透、双材料界面。先建立可重现数值证据，再扩 300–1,000 条 pilot；pilot 稳定后按数据学习曲线扩到 10k/50k，而非一开始生成 100k。

数据分层：

1. 单本构、真实尺度、可控作用，保留旧数据连续性。
2. 2–3 区域：软内芯+硬壳、布/皮+填充物、软体+硬件嵌件。
3. 薄壳厚度、连接/滑移、摩擦和界面条件显式标注。空瓶不填成实心，钱包不默认为均质块。
4. 在能控制 E/nu/rho 的前提下，独立随机化 appearance、damping、contact；避免固定阻尼成为材质类型捷径。
5. 保存 float32 位移/速度、应力/应变、力、坐标、单位、时间与相机。原始物理字段独立于渲染器；彩色图只作展示。
6. 部件语义/外观贴图只是资产候选，不自动等于真实内部材质。不可见结构保留多个候选或不确定性。

libuipc 接触鲁棒性不等于模型已标定或完整可微；多材料示例也不等于已验证粘接界面。断裂、损伤和脱粘放到专门阶段，不用简单粒子分离冒充真实裂纹。

## 10. A800 排程、资源与记录

- 开发先用 1 卡 profiling；正式 C/D 或 E/F 各占 4 卡可并行同预算实验，稳定后使用 8 卡训练单个更大模型。当前脚本通过 CUDA_VISIBLE_DEVICES/GPUS 控制，不默认常驻占满 GPU。
- 环境和大型数据在 AFS；本地 IO cache 使用已核对空间的可回收目录，保留原始数据和完整校验回执。G1b 已按用户继续优化的授权迁移 14.4 GiB 缓存，见 `G1_FEEDBACK_AND_G1B_PLAN_ZH.md`。
- 不预估未经测量的 epoch 小时数。先 profile 100 steps，记录 samples/s、peak memory、GPU utilization、data wait、通信与 backbone 耗时，再给完整 wall-clock 预算。
- 检查点保存 optimizer、epoch/step、per-rank RNG、manifest hash、train 统计和 config。当前 exact resume 仅支持相同 world size 的 epoch 边界；中断于 epoch 中间的权重只能作显式初始化，不伪称精确恢复。
- 每个实验新 output 目录，禁止静默覆盖 last.pt；结果保留逐样本记录和失败样本。

当前运行仅为少步软件/真实数据验收，不占用长训；是否提升需完整数据与受控训练结果。

## 11. 交付状态和下一批具体任务

完成：独立训练框架、真实数据兼容、数据/标签隔离、反馈控制、强主干适配器、DDP 入口和关键测试。精确验收结果见 `BUILD_STATUS.json`，不要将 tiny/small smoke 当作材料性能。

待执行顺序：

1. 完整数据审计与原文 split 核查；先用 DINOv2 控制组跑 C/D 各 1 seed 的 pilot。
2. 补原始响应/阻尼标定，确认旧数据对新目标的可信范围。
3. 在合法权重就绪后做 DINOv3-L/H+ 同条件比较，再决定最终主干。
4. 真正集成 VoMP，完成 E/F 与静态先验冲突实验。
5. PokeFlex/PhysTwin 的新动作验证，以及等预算 TTO。
6. 标定通过后才启动复合材料新数据 pilot。

这一顺序能够先处理 review 的因果证据和数据可信度，再把工作推向“真实视频 → 可用物理参数”，每一步都有明确的失败出口。

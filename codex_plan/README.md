# Claude 接手入口：DeformFieldBench / ICLR 核心 rebuttal

交接日期：**2026-11-09，Asia/Shanghai**。本文件用于迁移上下文，不代表远程任务已经恢复或任何计划实验已经完成。建议 Claude 首先读取本文件，再按下方顺序读取随包资料。

## 1. 用户当前要求与不可遗漏的约束

- **近期只完成三位 reviewer 的核心 rebuttal。** 用户给出的截止日期为 **2026-11-15**，具体时刻未确认；内部希望 11 月 14 日封版证据。日期目标以资源可用为前提，不能虚构完成状态。
- **目前没有空闲机器。** 用户最后明确说：“我目前还没有空闲的机器，先帮我继续优化方案”。现在只能做本地研究、审计、文稿与实现准备；不要 SSH 扫描、恢复训练/仿真、下载大权重或启动监控轮询。迁移给 Claude 不等于解除暂停。
- **如果严格证据表明当前系统没有实际提升，要立即通知用户，随后开始新方法 version 2 构建。** 条件性转向已授权，无需再问是否创建 V2；计算仍受当时的资源限制。不要等 V2 成功后才报告 V1 负结果。
- 不能把“小样本单种子没赢”“CI 跨零”“反馈无增益”混同为“整个系统已证伪”。区分反馈、辅助监督和完整系统的贡献。
- 每个阶段给用户实质反馈；loss/验证指标停滞按相应阶段规则早停，不无限续跑。不要创建每小时自动任务。
- 用户要求代码/项目文档及时提交到 `https://github.com/3365538768/DeformFieldBench`。本机账户 3365538768 就是用户所称 linrui，Git 作者 `linrui <3365538768@users.noreply.github.com>`。不要再次纠结账户名称。
- 不替用户发布 OpenReview 回复。先准备可审核的结果与回复稿。
- MLE 继续在 4090/5090 和旧目录执行；不要修改其数据、代理、任务或监控。FieldBench 的历史目标机器是 A800-2。

## 2. 路径、版本与资料优先级

| 位置 | 用途 / 注意事项 |
|---|---|
| 本交接包所在 `rsi_real_test/codex_plan/` | 自包含的目标、结论、文献索引、计划和审计快照；不含数据集/模型权重 |
| 本机 `C:\Users\shoul\Documents\ChatGPT\ICLR 2\.repo-sync\DeformFieldBench` | **最近实际提交/推送的权威 Git checkout**；交接前 commit `de6ad15545de68bdaa636dbffddf667e92fe210b` |
| 本机 `C:\Users\shoul\Documents\ChatGPT\ICLR 2\rsi_real_test` | 工作副本，部分文件早于权威 checkout；不要假定它与 GitHub master 同步，更不要盲目覆盖其未核实内容 |
| GitHub 仓库根目录 | 已直接是独立 FieldBridge 项目，不需要再进入名为 rsi_real_test 的子目录 |
| 旧论文源码 commit `131c0b063d81b5413691e43ea78fe3ba64e53e9c` | 老代码按用户要求已从当前根目录删除；用 `git show COMMIT:path` 阅读，勿覆盖当前实现 |
| A800-2（暂停） | `ssh root@180.153.172.11 -p 50002`；项目 `/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`；本轮没有连接核验 |
| `D:\ICLR\19436_Physical_Field_Bridged_M.pdf` | 本地论文 PDF，共 27 页；见 provenance 哈希；与各次 review 版本的对应仍需核实 |
| `D:\ICLR\research\paper_extracted.txt` | 既有提取文本；关键 Table 1/7、loss、真实 replay 已从 PDF 再次核对 |
| 本机 `.research_nf/` | 公开论文/README/arXiv API 的研究缓存，不在 Git；不要把整篇版权文本提交仓库 |

历史 SSH 配置、认证、代理和 HF token 都不在本交接包。用户说 A800 可使用 4090 同类代理；这只是既有意图，不代表旧代理现在仍可用。DINOv3 已获权限，历史 G1F 已训练；新执行时不要把 gated 认证视为尚未申请，也不要泄漏凭据。大权重/数据过去允许在新机器重下；当前无机器期间不下载。

**原始资源：**

- 代码：<https://github.com/3365538768/DeformFieldBench>
- 数据 ZIP：<https://huggingface.co/datasets/HandsomeHusky/DeformFieldBench>
- 同名 model repo：<https://huggingface.co/HandsomeHusky/DeformFieldBench>，不要与 dataset repo 混淆。
- 原 Gaussian 几何资源：<https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000>；用户已指出这里含几何，勿重新问同一问题。文件的实际下载/完整性仍需查回执。

## 3. 长期目标与这次 rebuttal 的边界

长期目标是从一段真实视频（静止或有形变）推断材料理解与可用的引擎参数，兼顾多引擎共享表征、少样本适配、物理中间量和真实拟合标签，尽量接近在线仿真拟合的行为效果，并明显强于无桥方法。

但**任意无标定视频唯一恢复真实 E/ν/ρ/yield 并迁移任意引擎不是一般可辨识的目标**：力、尺度、密度、内部结构、边界和阻尼都可能混淆。更合理的是在声明的观测/载荷/尺度条件下预测有效参数或响应分布，并在未见动作上验证。

这次短期不能把改用 Qwen、重做新引擎或新数据集的进展当成原论文桥接机制已经成立的证据。先回答原稿的关键质疑，严格结果不支持就及时转向。

## 4. 三位 reviewer 的核心意见

三位均为 weak reject。详细逐项方案与回复工作稿已随包复制；无需重新泛泛调研一遍再开始。

| Reviewer | 最重要的质疑 | 直接需要的证据 |
|---|---|---|
| a1nG | 材参预测提升不明确，散点像 GT 与预测无关 | 全量散点、族内相关/方差、常数控制、各参数与尾部、真实收益大小 |
| AKYF | 原仿真无标定，回弹 <5% 与 Spring-Gaus >60% 差异，缺独立 baseline | 原 MPM 物理校验与可靠范围；PIXIE/VoMP 等同条件比较；严格反馈-only 重训；训练阶段配置 |
| Nxg5 | Eq19–21 相关性不是一般物理规律，机制证据混杂，VLM 与消融不可复现 | 核对实际 loss 启用；loss-off/分阶段说明；分清场质量、推理依赖和重训收益；完整 VLM 与控制配置 |

两份用户 review 附件字节相同，SHA256 为 `7640e55006b41ad436ca2cc73121e98655426cd135f9dc55efde3e3d77dcd6a6`。无需在公共 Git 里上传 review 全文。最新本机附件路径：`C:\Users\shoul\.codex\attachments\4a40b46a-8c68-45df-bd8a-2103de1f31d2\已粘贴的文本.txt`。

## 5. 已核实结果：不能选择性报喜

### 5.1 本地论文（不是本轮新跑）

| 指标，越低越好 | Supervised | Ours |
|---|---:|---:|
| E log-space MedAE | 1.901 | 1.521 |
| ν MedAE | 0.048 | 0.045 |
| density log-space MedAE | 0.408 | 0.389 |
| yield log-space MedAE | 1.797 | 1.585 |
| 全样本 SPE 中位数 | 0.304 | 0.299 |
| 全样本 SPE 均值 | **0.314** | 0.334 |

各参数中位数较好，但总 SPE 中位数只改善约 1.64%，均值反而差约 6.37%。不能称全面胜出。Table 2 同引擎 replay 36.82 vs 23.86 dB、SSIM .868 vs .725 值得追查，但同生成器重放不验证真实物性，实际协议/预测产物仍缺。

Table 7 的已有消融没有独立隔离 feedback。方法各自剔除最差20%的指标不得作为主胜出证据。原 SPE 使用测试集 robust range，应披露；新 train-only 归一化指标另列，不悄悄换口径。

论文已有 5 段真实视频 replay（钱包/塑料瓶），15.864–20.561 dB、SSIM .7714–.8430；不是完全没有真实实验，但无真实材参 GT，无法证明绝对参数准确。附加物体、外部 benchmark 和 view sensitivity 都有历史表，但实际回执需核实。

### 5.2 当前重构项目的诊断（截至 11 月 8 日）

| 实验 | 主要结果 | 能/不能说明什么 |
|---|---|---|
| M2，512 train/128 val、seed17、3.574839M 参数 | C_aux .68281051，D_fields .68671702，B_rgb .69952394，G_features .70628557 | D 比 C 误差高 .57%；场被用到了但没比匹配无桥好。单 seed 验证诊断，不是原模型严格复现 |
| G1F，DINOv3 ViT-L/16 | B .805450，D .802526，G .802460，GT oracle .690323，shuffled .801946 | 真值场有潜在信息，预测场独立收益未建立。oracle 是测试特权诊断，不是部署能力或绝对上界 |
| NF，严格无场、512/128、seed17 | N0视频 .766416，N1首帧先验 .772903，N2后验 .779855 | 当前先验没有提升；95% nominal interval 覆盖89.77%，未校准。不同协议不能与 M2 排名 |
| G2，Warp/UIPC，324夹具案例，3 seeds | native .036906±.00459；response .036490±.00637；alignment .036421±.00645 | 改善小于种子波动，尚无可靠共享表征优势；输入是完整节点响应和已知载荷，不是 RGB |
| G2 seed17 未见 Z 载荷 replay | alignment约1.53%/1.14%，native约1.91%/1.79%；经典 inverse FEM约.0073%/.0343% | 距理想已知观测的经典求逆仍远；不能宣称已接近现实在线拟合 |
| G3 | 旧载荷理想位移仅 .02–.09px；动态软材料更细步比较约6.85%，未通过5%时间收敛；存在静止尾帧 | 激励、可见性和数值耗散是数据瓶颈；更大 backbone 不会自动修复 |

JSON 原始审计快照在 `reference/docs/`，详见清单。旧文档含历史“进行中/已暂停/旧机”等记录；以本文最新的**全部资源暂停、没有新训练**为准，不把旧 PID 当今天仍存活。最新远端释放核验是历史回执，接手前没有重新 SSH 验证。

## 6. 原实现审计发现与缺失证据

已读取固定旧 commit 的配置、loss、train、model2、消融与 VLM wrapper。可复跑审计脚本在 `reference/scripts/`，它必须在带 Git 历史和 docs 的正式仓库中运行，不在纯资料包里直接运行。

- 默认 full JSON：3×64 帧、224²、场112²、frozen DINOv2 `vitg14_reg`，320 epoch/batch1/lr1e-5；阶段100 flow-force、80 stress、100 joint、余40 param。**这是默认配置，不是论文实际训练回执**。
- 公开默认 `use_phys_loss=false, lambda_phys=0`；论文 Appendix 却写入 consistency objective。需查 checkpoint/config/log，不能猜测实际使用与否。
- loss 对通道绝对值均值后，把 V/T/H/W 联合归一化与相关；不是逐时刻物理平衡。恢复振荡的 reviewer 反例成立；没有证据就移除/降格经验正则，不称普适机械定律。
- forward wrapper 可传入 GT object mask，训练还支持可选 mixed GT/pred fields；默认 GT ratio=0，不等于实际运行无特权。核对推断输入契约，不在未核实时指控泄漏。
- 存在可选 shared decoder latent 到参数头旁路。**预测场图反馈**与**全部 decoder 特征反馈**是不同问题，控制定义要固定，不能混叫无桥。
- 默认 eval_split=train / eval_every_epochs=9999，VLM 默认 wrapper `qwen3.5-plus`，但 registry 有多个模型。这些不能证明论文实际选模或 VLM 身份。
- 原数据扫描4914有效pack、108对象与论文5000/100不一致，测试数量也有口径差异。要列 ID 和排除理由，不静默修 split。
- 每子步 damping 可能导致强耗散。倍率 .999、dt1e-4 对应连续衰减率约10.005/s；这是线索，不是唯一根因。dt 收敛比较需固定物理衰减率，再换算每步倍率。

**缺失且优先寻找的小文件：**原稿与检查点对应、SHA、resolved runtime config、阶段日志、逐样本预测、本构/参数mask、原split与对象映射、replay帧范围与nuisance、原sim完整有效配置、脱敏VLM请求/响应元数据。先复用这些，无需首先下载或重新生成全量数据。

## 7. 恢复算力后的最小执行顺序

当前只准备，不执行：

1. **复算与数据完整性**：原检查点单样本核对，再一次 val；若已有预测直接本地复算散点/常数/族内相关/尾部。复算不符先查版本，禁止先换架构。
2. **严格 C/D**：C 保留原 bottleneck、场预测、场 GT 监督、阶段与预算，仅去除待研究的预测场反馈信息；D 使用反馈。不是 detach，不是测试时 mask，不是去掉整个 bottleneck。明确 shared-latent 旁路保留/去除的定义。
3. **省算力但不偷换问题**：只共享反馈尚未影响训练的相同阶段；joint 受反馈影响必须分开。只训最后 head 是固定表征诊断，不能叫完整系统重训。目标 seeds17/29/43，资源不足报实际完成数，不以 bootstrap 代替训练重复。
4. **原 MPM 校验**：自由落体→小应变材料响应→release/drop；固定物理单位、阻尼率、初态/接触，比较 dt/分辨率。新 FEM 校准不替代旧 MPM 校验。不为凑60%回弹调整到“看起来正确”。
5. **外部 baseline**：PIXIE/VoMP公开权重各一个 val资产 smoke→冻结输入/映射/超参→预定完整测试。AutoPhyX先查checkpoint；OmniPhysGS小规模生成/行为比较并披露目标差异。无 GT 的参数填N/A，不帮baseline填GT。
6. **主测试与回复**：锁定协议后一次执行；对象级配对CI、各seed、各参数、尾部/失败和行为。test不选checkpoint、不挑对象、不据test反复调参。只将完成的证据填入工作稿。

主比较区分严格 RGB-only 和额外几何辅助；所有方法相同允许观测、初态、nuisance、相机与渲染。真实 replay 统一可见mask、手遮挡排除率与动态区域误差，没有独立参数GT就只谈有效行为。

## 8. 止损与 V2

预登记实际意义门槛沿用项目 **2% relative error improvement**，这是工程决策阈值，不是自然常数或显著性阈值。改善定义 `(control−candidate)/control`。对象不确定性和训练seed不确定性均报告，不能把seed×object全部当独立样本。

- C/D稳定实际改善：完成核心rebuttal，限定适用范围，不扩大V1。
- CI宽、种子反向：立即报告证据不足；预算上限到达就停止扩展，不宣称零效应。
- 严格审计合格、多seed下改善区间上界低于2%，或D明确劣于C：立即报告在该协议无实际反馈收益，停止为V1反馈投入，启动V2。
- 若C仍强于B，保留辅助监督价值；若各组近常数且排除工程问题，则更直接转向数据/表征。
- 原数据校验失败/原稿产物无法恢复：如实报告与修订；不能据此证明架构在所有可靠数据无效。

V2候选主线：**校准数据＋动态证据数值头＋训练期物理教师＋共享响应/引擎adapter**。先纯视频与常数/辅助/普通蒸馏/物理蒸馏严格对照，不强制测试时dense stress反馈；Qwen作为可移除语义先验，经反语义与同外观不同物性样本验证后才升级。真实拟合标签是带误差/来源的有效引擎参数，动作A拟合、B验收；未见引擎K-shot按独立对象/物理实验计，不按帧计。

## 9. 已研究的重要近邻：创新边界

完整细读记录在随包3份研究报告及其来源JSON，以下为接手时必须知道的结论，论文自报数字不能直接与本项目跨协议排名：

| 工作 | 对本项目的意义与比较边界 |
|---|---|
| [PIXIE](https://arxiv.org/html/2508.17437v2) | 3D多视角CLIP/体素→E/ν/ρ，静态资产先验，非首帧RGB即用；有公开权重/代码 |
| [VoMP](https://arxiv.org/html/2510.22975v2) | GeometryTransformer＋MatVAE三维物性预测，公开1.0/1.1；论文约3.59s从已有资产出发，不含手机重建全成本 |
| [AutoPhyX](https://arxiv.org/html/2610.29517v1) | 文本FiLM＋OpenCLIP体素；有代码，最近核验未找到明确checkpoint下载入口，不能断言永远未公开；不能输入GT材质答案文字 |
| [OmniPhysGS](https://arxiv.org/html/2501.18982v1) | 3DGS＋文本/视频扩散SDS逐场景优化本构专家，偏合理动态生成；不是天然目标视频参数求逆上界 |
| [ReconPhys](https://arxiv.org/html/2604.07882v1) | 单目20帧前馈联合GS与弹簧物理；有公开资源。spring k≠Pa单位E，按原生输出/未来行为比较 |
| [Analytic Dynamics](https://arxiv.org/html/2608.31025v1) | 已有位置/位移/F物理教师到视频学生蒸馏；jelly未见物体logE误差.265→.175。只“加中间物理量”不够新，不代表本项目dense反馈必有效 |
| [UniPixie](https://arxiv.org/html/2606.05399v2) | 已覆盖共享表征、多求解器概率输出与离线拟合蒸馏；多引擎adapter本身不够新，要证明未见引擎少样本响应迁移 |
| [PhysTwin](https://arxiv.org/html/2503.17973v1) / [EMPM](https://arxiv.org/html/2601.17251v1) | 真实视频拟合工具/比较来源，通常有RGBD/几何/控制等输入；优化标签不等于真实材参GT，成本和信息权限需公平 |
| [AdaptiGraph](https://arxiv.org/html/2407.07889v1) | 少量真实交互物性适配已有先例；RGBD、已知类别和物性变量条件不同，不能声称few-shot idea首次 |
| [PhysObjects](https://arxiv.org/html/2309.02561v4) | 可用作VLM材料语义/物理概念训练；分类/相对属性不是绝对连续力学参数测量 |

其他数据/引擎评估（PokeFlex、Deform360、PhysReal、Warp、libuipc、Newton等）及Qwen模型版本/资源见四方向报告和早期诊断。PokeFlex的N/m刚度不能直接当E；外观素材属性、拟合参数、实测参数必须带来源和单位区分。

## 10. 推荐阅读顺序与接手动作

1. 本文件，随后 `reference/docs/REBUTTAL_CORE_AND_V2_DECISION_ZH.md`。
2. `reference/docs/REBUTTAL_COMPUTE_MINIMAL_PROTOCOL_ZH.md`、`REBUTTAL_EXECUTION_PLAN_2026-11-09.json`：当前暂停和待做项目。
3. `reference/docs/REBUTTAL_PRIORITY_PLAN_2026-11-09_ZH.md`、`REBUTTAL_RESPONSES_WORKING_DRAFT.md`：详细证据逻辑与回复工作稿，工作稿**不可直接提交**。
4. M2/G1F/NF/G2/G3 JSON与解释，优先看协议/输入条件，勿只摘数字。
5. `FOUR_IDEAS_RESEARCH_AND_PLAN_2026-11-08_ZH.md`、`RESEARCH_FEASIBILITY_AND_METHOD_2026-11-08_ZH.md` 与来源JSON：V2创新空间和文献结果。

**接手第一条反馈建议**：说明已读取交接、当前只继续本地方案；列尚缺原稿回执而不重新问已知链接/机器身份。用户没有要求自动启动新任务或联系其他session，不主动发送跨会话消息。

交接包 `handoff_state.json` 保存机器可读状态，`MANIFEST.json` 保存每个文件的SHA256。引用 `reference/docs/` 文档里的 `../src` 或数据目录时，指的是正式项目，不是本资料包；不应把资料包当成可训练仓库。公开论文全文、权重、数据和密钥均未随包复制。

# 新 A800 迁移记录

目标项目目录：`/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`。
MLE、通用工具及配置复制到同一新盘的 `linrui/` 下。用户已决定 MLE 继续在旧 4090/5090 运行，由另一 session 管理；本迁移不改变旧目录、运行配置或监控。
MLE 活跃文件的新盘副本仅为复制时快照，不保证是一个同时刻的一致检查点，也不用于自动恢复。
旧盘保留，复制完成后仍需完整性校验、路径修复和运行环境验收。

## 2026-11-08 网络与代理验收

在新 A800 上实测三条代理路径：旧工作区的私有 HTTP 代理、新机已有的本地代理、Windows SSH 反向转发。
三条路径访问以下九个目标均返回 HTTP 200：

- Hugging Face 公共模型元数据、已授权账户身份、DINOv3 gated 配置。
- GitHub 仓库页面及 Release 重定向。
- PyPI、PyTorch CUDA wheel 索引。
- 现有 OpenAI 兼容 API 和 Anthropic 兼容 API 的模型列表。

另外，RSI 专用的两套 API 凭据、DINOv3 大权重的 HEAD/重定向、GitHub Git 读取协议和现有 27987 转发均通过。
API 检查仅覆盖认证与模型列表，未调用付费生成；权重 HEAD 也不代表完整权重已经下载校验。
新机直连 Hugging Face 和模型 API 超时，不能依赖直连。

HF 与模型 API 使用已验证的私有 HTTP 代理；PyPI 大 wheel 实测新机本地代理更快，依赖安装使用本地代理。
Windows 转发依赖本机和 SSH 会话在线，不作为无人值守下载的首选。
代理和令牌保存在新盘的私有文件中，权限为 0600，不进入代码仓库。

环境入口位于新盘 `linrui/fieldbridge-env.sh`，设置代理、HF token 路径和新盘缓存。
`scripts/with_proxy.py` 支持 `FIELDBRIDGE_PROXY_CONFIG`，HF 下载支持 `HF_HOME` / `HF_TOKEN_PATH`。
监控支持 `--host a800-2 --remote-root /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`。

新机原有 CUDA 12.6 兼容库必须加入本项目的 `LD_LIBRARY_PATH`。
系统 PyTorch 2.6、独立 PyTorch 2.8+cu126 GPU 运算均通过。pyuipc 0.0.29 和 Warp 1.11.1 独立环境已安装并完成真实 GPU 求解预检；加入路径迁移契约后，新机完整测试 57 项通过（8 条已有 fork 警告）。

G3r 在新机复用经过源文件/轨迹 SHA、图像解码与 flow 数值检查的片段，补齐至 32/32。最终 UIPC 校验后阶段完成；仍仅为诊断块体和图像空间遮挡，不代表真实三维工具接触已经实现。
显式 CPU 队列任务不再因其他用户占用 GPU 而阻塞；需 GPU 的验收单独执行并保留原有进程。

## 数据和权重

原始 41,425,509,436 字节数据 ZIP 在新机重新分块下载，完成后必须核对既有官方 SHA256，并检查外层与 NPZ 内层 CRC/数组模式。派生数据、几何、数值轨迹、训练检查点和历史结果继续复制；复制与校验分别登记，不能把 rsync 退出成功当作完整迁移验收。

DINOv2-S/G 在新机重新下载，完成 ZIP CRC 检查并与原部署 SHA256 比对。DINOv3 的 HF 权重在“不搬大权重”指令前已复制完成；新机仍须核对官方 LFS SHA、严格加载及 HF/原生特征一致性后才可用于训练。

新机默认 fused attention 的 HF/原生差约 1.55e-3，未通过原门槛。改为相同 FP32 math SDPA、禁用 TF32 后，最大差为 4.7311e-6，平均差为 3.4165e-7，原门槛不变；转换后的文件 SHA 与原部署完全一致。脚本已固定并记录 `fp32_no_tf32_math_sdpa_v1` 协议，用于权重转换验收，不将此值当作训练性能改善。

历史清单和统计量保留原始字节及哈希。`FIELDBRIDGE_PATH_MAP` 可指定私有运行配置 JSON，将旧绝对目录映射到新目录；匹配完整目录边界，最具体规则优先，映射记录进入训练 session 的 provenance。未设置时保持原行为。缓存索引仍需满足原清单哈希，不自动重写统计量或接受不匹配缓存。
迁移后的绝对路径和旧 `/tmp` 缓存软链接必须单独验收；未通过时不启动依赖它们的训练。

## MLE 数据复制范围

依据用户指定的五个任务清单，仅复制以下任务的数据集：

- `alaska2-image-steganalysis`
- `bms-molecular-translation`
- `hms-harmful-brain-activity-classification`
- `rsna-2022-cervical-spine-fracture-detection`
- `siim-isic-melanoma-classification`

第五个任务原始数据与验证划分位于另一处 AFS，由可访问的 5090 节点提供。
数据复制使用有目录限制的临时只读通道，迁移后撤销。
历史运行记录中的路径保留为证据；运行配置的路径映射须单独记录，不能据此自动恢复 MLE。

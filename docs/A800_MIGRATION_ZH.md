# 新 A800 迁移记录

目标项目目录：`/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`。
MLE、通用工具及配置位于同一新盘的 `linrui/` 下；MLE 保持用户暂停状态。
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

默认继续使用已验证的私有 HTTP 代理；新机本地代理可作备用。
Windows 转发依赖本机和 SSH 会话在线，不作为无人值守下载的首选。
代理和令牌保存在新盘的私有文件中，权限为 0600，不进入代码仓库。

环境入口位于新盘 `linrui/fieldbridge-env.sh`，设置代理、HF token 路径和新盘缓存。
`scripts/with_proxy.py` 支持 `FIELDBRIDGE_PROXY_CONFIG`，HF 下载支持 `HF_HOME` / `HF_TOKEN_PATH`。
监控支持 `--host a800-2 --remote-root /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`。

新机原有 CUDA 12.6 兼容库必须加入本项目的 `LD_LIBRARY_PATH`。
使用系统 PyTorch 2.6 的小矩阵 GPU 运算已通过；项目独立 PyTorch 2.8 和两套引擎环境仍需完成安装、验收。

## MLE 数据复制范围

依据用户暂停审计记录，仅保留以下五个任务的数据集：

- `alaska2-image-steganalysis`
- `bms-molecular-translation`
- `hms-harmful-brain-activity-classification`
- `rsna-2022-cervical-spine-fracture-detection`
- `siim-isic-melanoma-classification`

第五个任务原始数据与验证划分位于另一处 AFS，由可访问的 5090 节点提供。
数据复制使用有目录限制的临时只读通道，迁移后撤销。
历史运行记录中的路径保留为证据；运行配置的路径映射须单独记录，不能据此自动恢复 MLE。

# G1f：授权 DINOv3 接入与匹配对照

2026-11-07 用户确认授权后直接执行；已删除此前每小时自动任务。本阶段由 A800 常驻实验队列执行，网页采集仍为每 15 秒一次，二者不同。

## 权重验证

- 模型：`facebook/dinov3-vitl16-pretrain-lvd1689m`，固定 HF revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`。
- 官方代码固定 `6876159a11b4df116f30f667f8c9888617df0751`，不修改上游源码。
- 1,212,559,808 字节 HF safetensors 的 LFS SHA256 校验通过，见 [授权与数值回执](DINOV3_PREPARATION.json)。令牌及代理仅存在主机私有配置，不进入仓库或日志。
- 依据 [Transformers 官方转换实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py) 逆向恢复原模型 key、合并 Q/K/V、恢复固定 key-bias mask 与 RoPE buffer；所有源权重必须消费，目标 state 必须严格加载。没有随机权重兜底。
- 同一随机 FP32 输入，两张 224×224 图像，官方 Hub 与 HF 模型的 196×1024 patch 特征最大差 `5.126e-6`、平均差 `3.360e-7`，通过 `atol=rtol=2e-4`。此项验证格式转换，不代表材质效果。
- 导出的 `.pth` 名称后八位来自实际文件 SHA256；不冒用官方 `.pth` 文件哈希。历史 `model.py` 保持不变，旧 DINOv2 缓存与检查点仍有效。

## 实验顺序与控制

1. 构建 512 train / 128 val 的独立 DINOv3 RGB 特征缓存；物体划分、16 帧×3 视角、224 分辨率及 seed 17 与前阶段一致。
2. 四条真实 train/val 样本检查 RGB/缓存预测差不超过 0.001 标准化单位。完成回执前不允许训练启动。
3. 全量缓存文件逐一 SHA256 校验后复制至 `/tmp/linrui_fieldbridge/g1f_dinov3l16`；AFS 原件保留。节点临时盘清空后需重新执行迁移。
4. 独立训练 DINOv3 RGB 基线 B。不能复用 DINOv2 的参数头，也不能用旧 RGB 验证误差作为本阶段桥的比较门槛。
5. B 完成后，D 预测场、G 等容量特征、O 特权真场、S 置乱真场从同一冻结 B 检查点开始。使用 G1e 的归一化门与零场参照，保持匹配容量、损失和预算。

联合阶段至少观察 6 轮、连续 4 次验证未改善 0.2% 时自动停止；预热最多 2 轮。单独/全局暂停均复用网页控制。G1f 不自动提升到全量训练，不访问测试集。单种子验证结果不能当作多种子结论或在线拟合对比。

## 已发现的效率问题

首次串行缓存抽样读取耗时 0.801 / 0.690 / 0.589 秒，编码等待 CPU 数据解压。恢复后 78 个新样本耗时约 72.7 秒；改为四路、有界、保持顺序的 CPU 预取后，观测 156 个新样本 / 38.36 秒，即 4.07 条/秒。样本及时间窗口不同，仅作为工程吞吐观测，不声称固定数据端到端训练加速。后续仍需依据 `timing.jsonl` 判断训练瓶颈。

预取最多保留四个待处理样本的 future，CUDA 仅在主线程执行；缓存时禁止随机 view dropout。已有样本直接跳过，缺失文件会重建。额外的 1.21 GB Torch Hub 文件复制是官方本地权重加载缓存，不是重复联网下载。

网页实测在第 163 条暂停并继续成功，新的 GPU 进程从已有缓存恢复。队列依赖检查阻止未完成/失败的缓存、不同骨干基线提前放行。

640 条缓存已全部完成，四条真实 train/val 样本的 RGB/缓存预测最大差为 0；28,038,716,160 字节缓存的 640 个文件逐一通过 SHA256，本地迁移耗时 41.34 秒。B 已由队列实际启动。最新测试 39 项通过（21.78 秒，8 条既有 fork 警告），Node 语法与 Git diff 检查通过。结果回执：[缓存](G1F_CACHE_STATUS.json)、[迁移](G1F_LOCAL_CACHE.json)、[启动验证](G1F_STARTUP.json)。

B 的前两次联合验证 MAE 为 0.86658 → 0.86111，尚不能评价骨干收益。初期 epoch 训练约 19.6 秒，其中数据等待约 16.9 秒；进一步移除 `collate` 对无需补齐的张量进行补零复制。四条真实缓存样本、六次交替顺序测试：旧/新 CPU 组批中位耗时 204.47 / 116.29 ms，所有输出逐键严格一致，组批环节约 1.76 倍；不是端到端训练加速。见 [组批实测](G1F_COLLATE_BENCHMARK.json)。正在运行的 B 在启动时已加载旧实现，后续新启动任务使用同值优化；这不改变模型、采样或损失。

## 复现入口

```bash
pip install -e '.[dev,dinov3]'
python scripts/with_proxy.py python scripts/prepare_backbone.py --family dinov3 \
  --revision 6876159a11b4df116f30f667f8c9888617df0751
# 已取得 HF 授权，且私有 HF token 文件已配置：
python scripts/with_proxy.py python scripts/prepare_dinov3_hf.py
python scripts/setup_g1f.py --register
python scripts/optimization_manager.py
```

`--register` 适用于已有 A800 队列，要求 G1e 完成且不重置历史状态。新克隆可按配置分别执行缓存、迁移、B，再执行 D/G/O/S；还需准备真实数据、统计、官方代码与授权权重。

## 研究边界

DINOv3 接入只检验视觉特征是否限制当前结果，不解决受力未知、旧可视化场缺少物理单位或跨引擎 GT 语义差异。后续数据仍需经过版本、本构映射、几何尺度、加载与边界条件标定；共享响应表征与引擎专属头方案见 [跨引擎设计](CROSS_ENGINE_MATERIAL_ADAPTER_ZH.md)。这些因素不在本轮骨干对照中同时改变。

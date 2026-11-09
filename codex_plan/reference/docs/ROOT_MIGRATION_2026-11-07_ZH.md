# 根目录迁移说明（2026-11-07）

按项目维护要求，当前分支删除原论文旧实现，将原 `rsi_real_test/` 内的文件直接作为仓库根目录。新目录为：

```text
DeformFieldBench/
  pyproject.toml
  README.md
  src/fieldbridge/
  configs/
  scripts/
  monitor/
  tests/
  docs/
```

旧 `logic_model/`、`my_model/`、`my_utils/`、`eval/`、`eval_abalation/`、`simulation/`、`utils/`、`vlm_benchmark/`、`pretrained/`、旧脚本及旧环境配置不再发布在当前树中。需要旧论文实现时，可查看历史提交 `131c0b063d81b5413691e43ea78fe3ba64e53e9c`；迁移不重写历史。

## 独立运行的含义

- `fieldbridge` 的模型、训练、评估、数据处理及软件测试均来自本仓库，不导入旧目录。
- CPU 最小训练/评估使用 `configs/smoke_cpu.json` 和本地生成的软件样本，无需旧代码、CUDA 或预训练权重。
- 真实训练仍需另行准备数据、仅训练集拟合的统计、官方骨干源代码和权重；下载/导入脚本与配置保留在本仓库。DINO 官方代码放 `external/`，不是被移除的旧论文代码。
- Web 观测台连接用户已配置的 SSH `a800` 和部署目录。历史 G0/G1 迁移队列需要相应已存在的实验检查点和回执；它不是下载仓库后自动出现的历史任务。
- 仿真接口 `src/fieldbridge/simulation.py` 保留，实际仿真后端及真实视频到仿真拟合的完整链路仍待实现。

已发布的旧队列计划移动到 `docs/examples/OPTIMIZATION_PLAN_A800_2026-11-07.json`，避免其被新克隆误作活动计划。运行时的 `docs/OPTIMIZATION_PLAN.json` 在部署中创建且不随 Git 发布。`prepare_next_stage.py` 不再读取旧的 `ACTIVE_FULL_CD_JOBS.json` 来解析固定基准配置。

## 已有环境迁移

新克隆后在仓库根目录执行 `pip install -e '.[dev]'`。曾在旧仓库嵌套目录执行 editable 安装的环境需重新运行此命令，以更新包路径；脚本相对路径统一以根目录为工作目录。

A800 本来就在 `/mnt/afs_engine/linrui/rsi_real_test` 以这套平铺结构部署，因此没有搬动其数据、权重、日志或训练输出，也不需要改变当前训练进程的工作目录。此次调整的是 Git 仓库结构。

## 验证结果

将新根目录的 Git 导出包解压到独立临时目录，确认不存在 `rsi_real_test/`、`logic_model/`、`my_model/`、`simulation/`、`vlm_benchmark/` 或 `upstream/`。在新虚拟环境中 editable 安装该导出包，复用 A800 已安装的第三方依赖，检查 `fieldbridge.__file__` 位于导出目录的 `src/fieldbridge/`。

14 项测试全部通过（24.26 秒）；随后按 README 流程生成软件样本、拟合训练集统计、CPU 训练 2 轮共 6 步，再加载 `best_material.pt` 独立评估，验证样本覆盖完整且无重复。记录见 [ROOT_MIGRATION_VALIDATION.json](ROOT_MIGRATION_VALIDATION.json)。此验证没有借用旧论文实现，也未将软件样本当作真实材质性能证据。

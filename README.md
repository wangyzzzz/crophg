# crophg

`crophg` 是围绕论文 Result 3 主线整理的最小研究库。

当前目标不是复刻旧仓库的全部实验，而是先把论文最终需要保留的分析边界切清楚：

- 内部分析：`3.1A`、`3.1B`、`3.2A`、`3.2B`、`3.2C`
- 对外开放：`3.3A`、`3.3B`、`3.4A`、`3.4B`

## 章节映射

- `3.1A`: `H-only` 部署损失（内部）
- `3.1B`: `G` 的补偿与 `G+FULLH` 互补（内部）
- `3.2A`: 四场景下 `G+H` 的有效窗口（内部）
- `3.2B`: `H-only` 下单一 VI 的跨场景变化（内部）
- `3.2C`: `G` 背景下单一 VI 的真实增量与变化（内部）
- `3.3A`: 最终模型定义与输入表示（公开）
- `3.3B`: 以 `H_FULL` 为 baseline 的最终模型预测表现与压缩收益（公开）
- `3.4A`: 相对 `H_FULL` 的生长推进建模时机比较（公开）
- `3.4B`: `anchor×VI` 选择演化（公开）

## 当前状态

当前仓库已完成：

- 最小包结构
- 章节注册表
- 公开/内部脚本边界
- 统一 CLI 骨架
- 四场景 split 的最小迁移入口
- `3.1A`、`3.1B`、`3.2A`、`3.2B`、`3.2C` 的可执行主体迁入
- `3.1A -> 3.2C` 的最小 smoke 验证打通
- `3.4A/3.4B` 的公开分析入口
- `3.2A/3.2B/3.2C` 的内部 formal analysis 入口
- `3.1A/3.1B` 的内部 formal analysis 入口

当前仍在继续完善：

- `3.1A/3.1B/3.2A/3.2B/3.2C` 的正式 analysis markdown 迁移与润色
- `3.2A/3.2B/3.2C` 的 formal analysis 已能基于 `CropHG` 自身输出独立落盘
- `3.1A/3.1B` 的 formal analysis 已能基于 `CropHG` 自身输出独立落盘
- 更系统的 smoke / regression tests
- 对外 README 与 Methods 风格说明的进一步收口

## 快速查看

```bash
python -m crophg.cli list
python -m crophg.cli list --public-only
python -m crophg.cli describe 3.3A
python -m crophg.cli describe 3.4A
```


## 最小 Smoke

以下命令已在本地 `PEG2P` 环境下完成最小验证：

```bash
source /Users/wangyuze/Documents/Codex/2026-04-29/conda/miniconda3/etc/profile.d/conda.sh
conda activate PEG2P
cd /Users/wangyuze/Desktop/NWAFU/crophg
PYTHONPATH=src python scripts/result_3_1a.py --print-spec
PYTHONPATH=src python scripts/result_3_1b.py --print-spec
PYTHONPATH=src python scripts/result_3_2a.py --print-spec
PYTHONPATH=src python scripts/result_3_2b.py --print-spec
PYTHONPATH=src python scripts/result_3_2c.py --print-spec
```

本地最小 smoke 结果目录示例：

- `3.1A`: `/private/tmp/crophg_smoke_result_3_1a`
- `3.1B`: `/private/tmp/crophg_smoke_result_3_1b`
- `3.2A`: `/private/tmp/crophg_smoke_result_3_2a_models`
- `3.2B`: `/private/tmp/crophg_smoke_result_3_2b`
- `3.2C`: `/private/tmp/crophg_smoke_result_3_2c`

说明：

- 当前 `CropHG` 运行时导入统一使用 `models...` / `crophg...`，不再依赖旧仓库的 `src.models...` 命名空间。
- markdown 输出已加入无 `tabulate` 依赖的 fallback，因此在 `PEG2P` 中可直接完成 `summary.md` 落盘。

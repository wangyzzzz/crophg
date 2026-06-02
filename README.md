# CropVIG

`PEG2P` 环境下仅保留三个公开模型入口。默认使用 `run` 子命令完整执行模型训练与预测；`analyze` 子命令仅用于汇总已有输出目录。

- `CropVIG-1`

  ```bash
  PYTHONPATH=src python scripts/cropvig_1.py run --config <config.yaml> --output-dir <experiment_dir>
  ```

- `CropVIG-2`

  ```bash
  PYTHONPATH=src python scripts/cropvig_2.py run --config <config.yaml> --output-dir <experiment_dir>
  ```

- `CropVIG-3`

  ```bash
  PYTHONPATH=src python scripts/cropvig_3.py run --config <config.yaml> --output-dir <experiment_dir>
  ```

安装为本地包后也可以直接使用：

```bash
cropvig-1 run --config <config.yaml> --output-dir <experiment_dir>
cropvig-2 run --config <config.yaml> --output-dir <experiment_dir>
cropvig-3 run --config <config.yaml> --output-dir <experiment_dir>
```

已有结果汇总：

```bash
PYTHONPATH=src python scripts/cropvig_1.py analyze --input-dir <experiment_dir> --output-dir <report_dir>
PYTHONPATH=src python scripts/cropvig_2.py analyze --input-dir <experiment_dir> --output-dir <report_dir>
PYTHONPATH=src python scripts/cropvig_3.py analyze --input-dir <experiment_dir> --output-dir <report_dir>
```

最小示例检查：

```bash
PYTHONPATH=src python examples/check_cropvig_entrypoints.py
```

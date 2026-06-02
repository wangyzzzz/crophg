# CropVIG

`PEG2P` 环境下仅保留三个公开模型入口：

- `CropVIG-1`

  ```bash
  PYTHONPATH=src python scripts/cropvig_1.py --input-dir <cropvig_1_dir> --output-dir <report_dir>
  ```

- `CropVIG-2`

  ```bash
  PYTHONPATH=src python scripts/cropvig_2.py --input-dir <cropvig_2_dir> --output-dir <report_dir>
  ```

- `CropVIG-3`

  ```bash
  PYTHONPATH=src python scripts/cropvig_3.py --input-dir <cropvig_3_dir> --output-dir <report_dir>
  ```

安装为本地包后也可以直接使用：

```bash
cropvig-1 --input-dir <cropvig_1_dir> --output-dir <report_dir>
cropvig-2 --input-dir <cropvig_2_dir> --output-dir <report_dir>
cropvig-3 --input-dir <cropvig_3_dir> --output-dir <report_dir>
```

最小示例检查：

```bash
PYTHONPATH=src python examples/check_cropvig_entrypoints.py
```

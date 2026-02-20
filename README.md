# DA-DeformMamba 贵州逐日降雨预测（2015-2024）

本项目提供一套可直接运行的时空降雨预测流水线，核心能力包括：
- 多城市多变量输入（空间维）
- 长时间窗建模（时间维）
- GMM 无监督子域划分
- 1D Deformable Attention 时间偏移对齐
- ConBiMamba 编码（未安装 `mamba-ssm` 时自动回退到 GRU）
- GRL 域对抗训练
- 点预测（MSE）与分位数预测（Pinball / Huber-Pinball）

## 项目结构

```text
.
├─configs/                     # 训练配置
├─scripts/
│  ├─train.py                  # 训练与评估主入口
│  ├─plot_results.py           # 训练曲线/指标图
│  └─plot_predictions.py       # 预测时序图
├─models/                      # 模型结构
├─runs/                        # 训练输出目录
├─data.py dataset.py ...       # 数据处理与损失/指标
└─README.md
```

## 1. 环境要求

## 1.1 推荐版本
- OS: Ubuntu 22.04
- Python: 3.10
- CUDA: 12.8
- PyTorch: 2.7.0 (cu128)

## 1.2 快速创建环境（Linux）

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 1.3 依赖安装策略

本仓库有两类依赖文件：
- `requirements.txt`: 通用依赖（跨平台）
- `requirements-lock.txt`: 某一台机器导出的锁版本（不一定跨平台）

建议：
1. 先按你的服务器环境安装兼容版本。
2. 安装成功后执行 `pip freeze > requirements-lock-linux-py310.txt`，形成你自己的锁文件。

## 1.4 PyTorch 安装示例（CUDA 12.8）

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
```

## 1.5 其余常用依赖示例（Python 3.10）

```bash
pip install numpy==2.2.6 pandas==2.2.3 scikit-learn==1.6.1 \
  PyYAML==6.0.2 einops==0.8.1 tqdm==4.67.1 matplotlib==3.9.4 openpyxl==3.1.5
```

安装后检查：

```bash
python -c "import torch, numpy, pandas; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__, pandas.__version__)"
```

## 2. 数据准备

`configs/gz_rain.yaml` 中的 `data_path` 必须与实际文件路径一致。

示例：
- 根目录：`20152024.xlsx`
- 或数据目录：`data/2015-2024贵州降水数据(1).xlsx`

可先检查文件是否存在：

```bash
ls -lah
ls -lah data
```

数据格式要求：
- 第一列是日期（可被解析为 `date`）
- 后续列为各城市逐日降水值（数值）

## 3. 训练

## 3.1 标准训练

```bash
python scripts/train.py --config configs/gz_rain.yaml
```

## 3.2 快速验证配置

```bash
python scripts/train.py --config configs/gz_rain_quick_all.yaml
```

## 3.3 断点续训

自动从 `runs/<exp_name>/<target_city>/last.ckpt` 继续：

```bash
python scripts/train.py --config configs/gz_rain.yaml --resume
```

指定 checkpoint 继续（仅单目标）：

```bash
python scripts/train.py --config configs/gz_rain.yaml --resume_path runs/<exp_name>/<target_city>/last.ckpt
```

## 3.4 训练脚本已支持的关键能力
- AMP：使用 `torch.amp.autocast` + `torch.amp.GradScaler`
- 完整 checkpoint：保存 `model/optimizer/scheduler/scaler/global_iter/rng_state/history`
- best 模型保存：`best.pt` 与 `best.ckpt`

## 4. 使用真正的 Mamba（禁用 GRU 回退）

若看到警告：
`mamba_ssm not found. Using GRU fallback.`
说明当前未启用真正的 Mamba。

## 4.1 安装编译依赖（Ubuntu）

```bash
apt update
apt install -y build-essential gcc g++ ninja-build
```

## 4.2 安装 `causal-conv1d` 与 `mamba-ssm`

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4

pip install -U pip setuptools wheel packaging
pip install -i https://pypi.org/simple --no-build-isolation causal-conv1d
pip install -i https://pypi.org/simple --no-build-isolation mamba-ssm
```

验证：

```bash
python -c "import mamba_ssm, causal_conv1d; print('mamba ok')"
```

## 5. 训练输出说明

按城市输出（`runs/<exp_name>/<city>/`）：
- `best.pt`: 最优模型权重（推理常用）
- `best.ckpt`: 最优检查点（含配置与最优分数）
- `last.ckpt`: 最近一次完整训练状态（用于续训）
- `predictions.csv`: 预测结果
- `history.csv`: 每轮训练日志
- `metrics.json`: 单城市指标

实验级汇总（`runs/<exp_name>/`）：
- `metrics_all.json`
- `predictions_all.csv`
- `history_all.csv`
- `plots/` 或你指定的 `figures/`

## 6. 可视化

## 6.1 训练曲线与指标图

```bash
python scripts/plot_results.py --exp_dir runs/gz_rain_da_deformmamba
```

可选参数：
- `--horizon`
- `--tail_points`
- `--cities`（逗号分隔城市名）

## 6.2 预测时序图

```bash
python scripts/plot_predictions.py \
  --pred_csv runs/gz_rain_da_deformmamba/predictions_all.csv \
  --out_dir runs/gz_rain_da_deformmamba/figures
```

## 6.3 中文乱码处理

如果图中中文乱码，先安装中文字体：

```bash
apt update && apt install -y fonts-noto-cjk
```

本仓库绘图脚本已自动尝试以下字体：
- `Noto Sans CJK SC`
- `Source Han Sans CN`
- `Microsoft YaHei`
- `SimHei`
- `WenQuanYi Zen Hei`
- `PingFang SC`

如仍乱码，清理 Matplotlib 缓存后重画：

```bash
python -c "import matplotlib as mpl, shutil; shutil.rmtree(mpl.get_cachedir(), ignore_errors=True)"
python scripts/plot_results.py --exp_dir runs/gz_rain_da_deformmamba
python scripts/plot_predictions.py --pred_csv runs/gz_rain_da_deformmamba/predictions_all.csv --out_dir runs/gz_rain_da_deformmamba/figures
```

注意：已生成的乱码图片不能直接修复，必须重新绘图。

## 7. 远程训练建议（防断线）

安装并使用 `tmux`：

```bash
apt update && apt install -y tmux
tmux new -s rain
source venv/bin/activate
python scripts/train.py --config configs/gz_rain.yaml
```

常用命令：
- 查看会话：`tmux ls`
- 进入会话：`tmux attach -t rain`
- 分离会话：`Ctrl+b` 后按 `d`

## 8. 成果打包与下载

## 8.1 服务器打包

```bash
cd ~/260219_ML-rainfall-time-prediction
tar -czvf rain_results_$(date +%Y%m%d_%H%M%S).tar.gz runs/gz_rain_da_deformmamba
```

建议连同配置与依赖一起打包：

```bash
tar -czvf rain_bundle_$(date +%Y%m%d_%H%M%S).tar.gz \
  runs/gz_rain_da_deformmamba \
  configs \
  requirements*.txt
```

## 8.2 下载到本地

```bash
scp root@<SERVER_IP>:~/260219_ML-rainfall-time-prediction/rain_bundle_*.tar.gz .
```

本地解压：

```bash
tar -xzvf rain_bundle_*.tar.gz
```

## 9. 常见问题（FAQ）

1. `ModuleNotFoundError: No module named 'utils'`
- 已在 `scripts/train.py` 内处理项目根路径导入。
- 旧版本可临时用：`PYTHONPATH=. python scripts/train.py --config ...`

2. `FileNotFoundError`（数据文件找不到）
- 检查配置里的 `data_path` 是否与真实路径一致。

3. `mamba-ssm` 安装失败 / `g++ not found`
- 先安装：`apt install -y build-essential gcc g++ ninja-build`

4. `plot_results.py` 报参数错误
- 必填参数是 `--exp_dir`。

5. `plot_predictions.py` 报参数错误
- 必填参数是 `--pred_csv` 和 `--out_dir`。

## 10. 复现建议

- 每次实验固定随机种子和配置文件。
- 训练前保存 Git commit id。
- 每次训练后保存：
  - `metrics_all.json`
  - `history_all.csv`
  - 当前依赖锁文件（如 `requirements-lock-linux-py310.txt`）


# Colab 免费 GPU 两城市强制 Mamba 执行清单

## 目标
- 在 Colab 免费 GPU 上完成两城市验证。
- 强制使用 Mamba（禁止 GRU fallback）。
- 生成并保存损失、预测、MAE、RMSE 可视化结果。

## 前置条件
- 仓库已包含：
  - `configs/gz_rain_quick_2cities_mamba.yaml`
  - `scripts/train.py`
  - `scripts/plot_results.py`
- 数据文件：`20152024.xlsx`

## Colab 执行步骤

### 1. 切换运行时到 GPU
- Colab 菜单：`Runtime -> Change runtime type -> GPU`

```python
!nvidia-smi
```

### 2. 挂载 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. 克隆代码并进入项目
```bash
%cd /content
!git clone <你的仓库URL> rainfall
%cd /content/rainfall
```

### 4. 复制数据到项目根目录
```bash
!cp "/content/drive/MyDrive/rainfall/20152024.xlsx" /content/rainfall/20152024.xlsx
!ls -lh /content/rainfall/20152024.xlsx
```

### 5. 安装依赖（CUDA + mamba-ssm）
```bash
!python -m pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -r requirements.txt
!pip install mamba-ssm
```

### 6. 环境检查（必须 True / True）
```python
import importlib.util, torch
print("cuda_available =", torch.cuda.is_available())
print("mamba_installed =", importlib.util.find_spec("mamba_ssm") is not None)
```

### 7. 启动训练（两城市，强制 Mamba）
```bash
%env PYTHONPATH=.
!python -u -m scripts.train --config configs/gz_rain_quick_2cities_mamba.yaml --device cuda 2>&1 | tee runs/train_2cities_mamba.log
```

### 8. 生成可视化图
```bash
!python scripts/plot_results.py --exp_dir runs/gz_rain_da_deformmamba_2cities_mamba --horizon 1 --tail_points 300
```

### 9. 结果保存到 Drive 并打包
```bash
!mkdir -p "/content/drive/MyDrive/rainfall_outputs"
!cp -r runs/gz_rain_da_deformmamba_2cities_mamba "/content/drive/MyDrive/rainfall_outputs/"
!tar -czvf /content/gz_rain_da_deformmamba_2cities_mamba.tgz -C runs gz_rain_da_deformmamba_2cities_mamba
```

## 验收标准
- 日志中不出现 `Using GRU fallback`。
- 存在：
  - `runs/gz_rain_da_deformmamba_2cities_mamba/metrics_all.json`
  - `runs/gz_rain_da_deformmamba_2cities_mamba/predictions_all.csv`
  - `runs/gz_rain_da_deformmamba_2cities_mamba/history_all.csv`
- 图像存在：
  - `runs/gz_rain_da_deformmamba_2cities_mamba/plots/loss_curve_*.png`
  - `runs/gz_rain_da_deformmamba_2cities_mamba/plots/prediction_*_h1.png`
  - `runs/gz_rain_da_deformmamba_2cities_mamba/plots/metric_bar_rmse_mae.png`

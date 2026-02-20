# DA-DeformMamba for Guizhou Daily Rainfall (2015–2024)

本仓库提供一套**可直接运行**的端到端代码，实现论文中的 DA-DeformMamba 思路：
- 多城市（空间）+ 多年逐日（时间）的多变量输入
- GMM 无监督子域划分（年代/气候模态）
- Deformable Attention（1D，可学习时间偏移）对齐动态传播滞后
- (Con)BiMamba 编码器（双向 Mamba；若未安装 mamba-ssm 则自动退化为 GRU 版本）
- GRL 域对抗（Domain Adversarial）实现域适应/域泛化
- 点预测（MSE）或分位数预测（Pinball / Huber-Pinball）

> 说明：你提供的 3_7_Deformable_Attention.py 里包含 1D/2D 版本，但 2D 版本在公开片段中存在维度不一致问题。本实现使用**1D 变形注意力**实现“时间滞后对齐”，空间（多城市）信息通过输入投影与后续编码器融合，训练/推理稳定可复现。

## 0. 环境准备
Python >= 3.9，建议使用 conda/venv。

安装依赖：
```bash
pip install -r requirements.txt
```

若你希望固定当前环境版本（推荐云上训练）：
```bash
pip install -r requirements-lock.txt
```

若你希望启用真正的 Mamba（而非 GRU 退化版本），请额外安装：
```bash
pip install mamba-ssm
```

## 1. 数据准备
将你的数据文件放到任意路径，例如：
`data/2015-2024贵州降水数据(1).xlsx`

表格要求：
- 第一列为日期（本仓库默认识别为 `Unnamed: 0` 或 `date`）
- 其余列为城市/州的逐日降雨（数值）

## 2. 训练
```bash
python scripts/train.py --config configs/gz_rain.yaml
```

断点续训（自动读取 `runs/<exp_name>/<target_city>/last.ckpt`）：
```bash
python scripts/train.py --config configs/gz_rain.yaml --resume
```

从指定 checkpoint 续训（仅单目标）：
```bash
python scripts/train.py --config configs/gz_rain.yaml --resume_path runs/<exp_name>/<target_city>/last.ckpt
```

你可以在 `configs/gz_rain.yaml` 中修改：
- target_city：预测的中心城市
- lookback L、horizon H
- 邻域城市选择方式（all / topk_corr）
- 训练/验证/测试按年份划分
- 是否做分位数预测、分位数集合等

## 3. 输出
- `runs/<exp_name>/best.pt`：最佳模型
- `runs/<exp_name>/metrics.json`：测试指标
- `runs/<exp_name>/predictions.csv`：预测与真实值（便于画图）
- `runs/<exp_name>/<target_city>/last.ckpt`：完整训练状态（model/optimizer/scaler/scheduler/随机状态）

## 4. 常见问题
- 若提示 `mamba_ssm` 缺失：不会影响运行，会自动用 GRU 代替；要与论文一致建议安装 `mamba-ssm`。
- 若你要“轮流预测所有城市”：在配置中设置 `target_city: "__all__"`，脚本会自动循环训练/测试并汇总结果（耗时更久）。

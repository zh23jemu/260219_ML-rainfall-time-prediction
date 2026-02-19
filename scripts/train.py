import os
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import set_seed, ensure_dir, save_json, to_numpy
from data import load_gz_rainfall_xlsx, add_time_features, split_by_year, StandardScaler
from dataset import SlidingWindowDataset
from gmm_domain import compute_window_stats, fit_gmm_bic, predict_domain
from models import DADeformMamba
from losses import mse_loss, quantile_loss, domain_ce_loss
from metrics import rmse, mae, picp, pinaw, aql

def compute_corr_order(X_train: np.ndarray, y_train: np.ndarray):
    # corr between each feature col and target
    corrs = []
    y = y_train - y_train.mean()
    denom_y = (y**2).sum() + 1e-6
    for j in range(X_train.shape[1]):
        x = X_train[:, j] - X_train[:, j].mean()
        denom_x = (x**2).sum() + 1e-6
        corrs.append(float((x*y).sum() / np.sqrt(denom_x * denom_y)))
    order = np.argsort(-np.abs(corrs))
    return order

def lambda_schedule(curr_iter, max_iter, gamma=10.0):
    p = min(curr_iter, max_iter) / float(max_iter)
    lamb = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    return float(lamb)

def run_one_target(cfg, df, target_city: str, device: str):
    L = int(cfg["lookback_L"])
    H = int(cfg["horizon_H"])
    use_time = bool(cfg["use_time_features"])
    neighbor_mode = cfg["neighbor_mode"]
    topk = int(cfg["topk"])

    # 城市列
    city_cols = [c for c in df.columns if c not in ["date", "doy_sin", "doy_cos"]]
    assert target_city in city_cols, f"target_city '{target_city}' not in columns: {city_cols}"

    # 特征矩阵（先不标准化）
    if use_time:
        feat_cols = city_cols + ["doy_sin", "doy_cos"]
    else:
        feat_cols = city_cols

    X_all = df[feat_cols].values.astype(np.float32)  # [T, V]
    y_all = df[target_city].values.astype(np.float32) # [T]

    # 划分
    train_years = cfg["split"]["train_years"]
    val_years = cfg["split"]["val_years"]
    test_years = cfg["split"]["test_years"]
    train_mask, val_mask, test_mask = split_by_year(df, train_years, val_years, test_years)

    # 仅在训练段上决定 neighbor 子集（如果 topk_corr）
    if neighbor_mode == "topk_corr":
        # 只在城市列上做相关排序，不包含时间特征
        X_city = df[city_cols].values.astype(np.float32)
        order = compute_corr_order(X_city[train_mask], y_all[train_mask])
        # 把 target_city 放在最后（可选），这里保持 order 但剔除 target 自身再取 topk
        target_idx = city_cols.index(target_city)
        order = [i for i in order if i != target_idx]
        keep_city_idx = order[:topk] + [target_idx]  # neighbors + target
        kept_cols = [city_cols[i] for i in keep_city_idx]
        if use_time:
            kept_cols = kept_cols + ["doy_sin", "doy_cos"]
        X_all = df[kept_cols].values.astype(np.float32)
        V_in = X_all.shape[1]
    else:
        V_in = X_all.shape[1]

    # 标准化（用训练段统计）
    scaler = StandardScaler().fit(X_all[train_mask])
    X_all_z = scaler.transform(X_all)

    # 计算 GMM 子域（在训练样本窗口上 fit）
    S_all = compute_window_stats(X_all_z, y_all, L=L, H=H)
    S_train = S_all[train_mask]
    gmm, k_best, bic = fit_gmm_bic(S_train, k_min=cfg["gmm"]["k_min"], k_max=cfg["gmm"]["k_max"],
                                  random_state=cfg["gmm"]["random_state"])
    domain_all = predict_domain(gmm, S_all)  # [-1 for invalid]
    num_domains = int(k_best)

    # 构造 Dataset（注意：domain 取 t 对应标签；t<L-1 或 t>T-H 为 -1）
    def make_loader(mask, shuffle):
        X = X_all_z[mask]
        y = y_all[mask]
        d = domain_all[mask]
        ds = SlidingWindowDataset(X, y, d, L=L, H=H)
        return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                          num_workers=cfg["train"]["num_workers"], drop_last=True)

    tr_loader = make_loader(train_mask, shuffle=True)
    va_loader = make_loader(val_mask, shuffle=False)
    te_loader = make_loader(test_mask, shuffle=False)

    # 模型
    task_mode = cfg["task"]["mode"]
    quantiles = cfg["task"]["quantiles"]
    model = DADeformMamba(
        V_in=V_in, L=L, H=H, num_domains=num_domains,
        d_model=cfg["model"]["d_model"],
        deform_layers=cfg["model"]["deform_layers"],
        deform_heads=cfg["model"]["deform_heads"],
        deform_groups=cfg["model"]["deform_groups"],
        deform_kernel=cfg["model"]["deform_kernel"],
        conbimamba_layers=cfg["model"]["conbimamba_layers"],
        dropout=cfg["model"]["dropout"],
        task_mode=task_mode,
        quantiles=quantiles,
        domain_hidden=cfg["domain_adv"]["domain_hidden"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler_amp = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["use_amp"]) and device.startswith("cuda"))

    beta = float(cfg["domain_adv"]["beta"]) if cfg["domain_adv"]["enabled"] else 0.0
    gamma = float(cfg["domain_adv"]["grl_gamma"])
    max_iter = int(cfg["domain_adv"]["max_iter"])

    use_huber = bool(cfg["task"]["use_huber_pinball"])
    delta = float(cfg["task"]["huber_delta"])

    best_val = 1e18
    best_state = None
    global_iter = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"[{target_city}] epoch {epoch}")
        for xb, yb, db in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)

            # domain labels：过滤掉 -1（无效），避免 CE 报错
            valid_domain = (db >= 0)
            if valid_domain.sum() == 0:
                continue

            lamb = lambda_schedule(global_iter, max_iter=max_iter, gamma=gamma) if cfg["domain_adv"]["enabled"] else 0.0

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler_amp.is_enabled()):
                pred, dom_logits, feat = model(xb, grl_lambda=lamb)

                if task_mode == "point":
                    task_loss = mse_loss(yb, pred)
                else:
                    task_loss = quantile_loss(yb, pred, quantiles, use_huber=use_huber, delta=delta)

                dom_loss = domain_ce_loss(dom_logits[valid_domain], db[valid_domain])
                loss = task_loss + beta * dom_loss

            scaler_amp.scale(loss).backward()
            if cfg["train"]["grad_clip"] is not None:
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler_amp.step(opt)
            scaler_amp.update()

            global_iter += 1
            pbar.set_postfix({"loss": float(loss.item()), "task": float(task_loss.item()), "dom": float(dom_loss.item()), "λ": lamb})

        # validation (use RMSE on median/point)
        val_rmse = evaluate(model, va_loader, device, task_mode, quantiles)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # test
    model.load_state_dict(best_state)
    test_metrics, pred_df = test_and_dump(model, te_loader, device, task_mode, quantiles, target_city)

    return {
        "target_city": target_city,
        "num_domains": num_domains,
        "best_val_rmse": float(best_val),
        **test_metrics
    }, pred_df, best_state, scaler, gmm

@torch.no_grad()
def evaluate(model, loader, device, task_mode, quantiles):
    model.eval()
    ys, ps = [], []
    for xb, yb, db in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred, _, _ = model(xb, grl_lambda=0.0)
        if task_mode == "point":
            p = pred
        else:
            # 用中位数分位数做点估计（tau=0.5）
            q = list(quantiles)
            mid = q.index(0.5) if 0.5 in q else len(q)//2
            p = pred[..., mid]
        ys.append(yb.cpu().numpy())
        ps.append(p.cpu().numpy())
    y = np.concatenate(ys, axis=0).reshape(-1)
    p = np.concatenate(ps, axis=0).reshape(-1)
    return rmse(y, p)

@torch.no_grad()
def test_and_dump(model, loader, device, task_mode, quantiles, target_city):
    model.eval()
    ys, p50s, los, his = [], [], [], []
    for xb, yb, db in loader:
        xb = xb.to(device)
        pred, _, _ = model(xb, grl_lambda=0.0)
        y = yb.numpy()

        if task_mode == "point":
            p = pred.cpu().numpy()
            p50 = p
            lo = hi = None
        else:
            p_np = pred.cpu().numpy()  # [B,H,Q]
            q = list(quantiles)
            mid = q.index(0.5) if 0.5 in q else len(q)//2
            p50 = p_np[..., mid]
            # interval: min/max quantile
            lo = p_np[..., 0]
            hi = p_np[..., -1]

        ys.append(y)
        p50s.append(p50)
        if lo is not None:
            los.append(lo); his.append(hi)

    y = np.concatenate(ys, axis=0)      # [N,H]
    p50 = np.concatenate(p50s, axis=0)  # [N,H]

    metrics = {
        "rmse": rmse(y.reshape(-1), p50.reshape(-1)),
        "mae": mae(y.reshape(-1), p50.reshape(-1)),
    }

    if task_mode != "point":
        lo = np.concatenate(los, axis=0)
        hi = np.concatenate(his, axis=0)
        metrics.update({
            "picp": picp(y.reshape(-1), lo.reshape(-1), hi.reshape(-1)),
            "pinaw": pinaw(lo.reshape(-1), hi.reshape(-1), y.reshape(-1)),
            "aql": aql(y.reshape(-1), lo.reshape(-1), hi.reshape(-1), alpha=1- (quantiles[-1]-quantiles[0])),
        })

    # dump csv
    rows = []
    for i in range(y.shape[0]):
        for h in range(y.shape[1]):
            row = {"sample": i, "horizon": h+1, "y_true": float(y[i,h]), "y_pred": float(p50[i,h]), "target_city": target_city}
            if task_mode != "point":
                row["y_lo"] = float(lo[i,h]); row["y_hi"] = float(hi[i,h])
            rows.append(row)
    pred_df = pd.DataFrame(rows)
    return metrics, pred_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["train"]["seed"]))

    # load data
    df = load_gz_rainfall_xlsx(cfg["data_path"])
    if cfg["use_time_features"]:
        df = add_time_features(df)

    exp_dir = os.path.join("runs", cfg["exp_name"])
    ensure_dir(exp_dir)

    target = cfg["target_city"]
    targets = []
    city_cols = [c for c in df.columns if c not in ["date", "doy_sin", "doy_cos"]]
    if target == "__all__":
        targets = city_cols
    else:
        targets = [target]

    all_metrics = []
    all_pred = []

    for tcity in targets:
        metrics, pred_df, best_state, scaler, gmm = run_one_target(cfg, df, tcity, args.device)
        all_metrics.append(metrics)
        all_pred.append(pred_df)

        # save model
        tdir = os.path.join(exp_dir, tcity)
        ensure_dir(tdir)
        torch.save(best_state, os.path.join(tdir, "best.pt"))
        pred_df.to_csv(os.path.join(tdir, "predictions.csv"), index=False)
        save_json(metrics, os.path.join(tdir, "metrics.json"))

    # summary
    save_json(all_metrics, os.path.join(exp_dir, "metrics_all.json"))
    pd.concat(all_pred, axis=0).to_csv(os.path.join(exp_dir, "predictions_all.csv"), index=False)
    print("Done. Results saved to:", exp_dir)

if __name__ == "__main__":
    main()

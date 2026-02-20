import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training and evaluation results.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory under runs/")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon to plot (1-based)")
    parser.add_argument("--tail_points", type=int, default=300, help="Tail points for prediction plots")
    parser.add_argument("--cities", type=str, default="", help="Optional comma-separated city list")
    return parser.parse_args()

def configure_matplotlib_for_chinese():
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in preferred_fonts if name in installed), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def resolve_cities(exp_dir: Path, cities_arg: str):
    if cities_arg.strip():
        return [c.strip() for c in cities_arg.split(",") if c.strip()]
    metrics_path = exp_dir / "metrics_all.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["target_city"] for item in data if "target_city" in item]
    city_dirs = [p.name for p in exp_dir.iterdir() if p.is_dir()]
    return sorted(city_dirs)


def plot_loss_curve(city: str, history_df: pd.DataFrame, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = history_df["epoch"].values
    ax1.plot(x, history_df["train_loss_mean"].values, label="train_loss", linewidth=1.8)
    ax1.plot(x, history_df["train_task_loss_mean"].values, label="task_loss", linewidth=1.4)
    ax1.plot(x, history_df["train_dom_loss_mean"].values, label="dom_loss", linewidth=1.4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, history_df["val_rmse"].values, color="tab:red", label="val_rmse", linewidth=1.8)
    ax2.set_ylabel("Validation RMSE")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"{city} - Loss and Val RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prediction(city: str, pred_df: pd.DataFrame, horizon: int, tail_points: int, out_path: Path):
    d = pred_df[pred_df["horizon"] == int(horizon)].copy()
    if tail_points is not None and tail_points > 0:
        d = d.tail(int(tail_points))
    d = d.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(d))
    ax.plot(x, d["y_true"].values, label="True", linewidth=1.7)
    ax.plot(x, d["y_pred"].values, label="Pred", linewidth=1.5)

    if "y_lo" in d.columns and "y_hi" in d.columns:
        ax.fill_between(x, d["y_lo"].values, d["y_hi"].values, alpha=0.2, label="PI")

    ax.set_title(f"{city} - Horizon {horizon}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Rainfall")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path):
    plot_df = metrics_df[["target_city", "rmse", "mae"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(plot_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], plot_df["rmse"].values, width=width, label="RMSE")
    ax.bar([i + width / 2 for i in x], plot_df["mae"].values, width=width, label="MAE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["target_city"].values, rotation=20)
    ax.set_ylabel("Value")
    ax.set_title("City-wise RMSE and MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    configure_matplotlib_for_chinese()
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cities = resolve_cities(exp_dir, args.cities)
    if not cities:
        raise RuntimeError("No cities found in experiment directory.")

    metrics_path = exp_dir / "metrics_all.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics_all.json: {metrics_path}")
    metrics_df = pd.DataFrame(json.load(open(metrics_path, "r", encoding="utf-8")))

    for city in cities:
        city_dir = exp_dir / city
        history_path = city_dir / "history.csv"
        pred_path = city_dir / "predictions.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history.csv: {history_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions.csv: {pred_path}")

        history_df = pd.read_csv(history_path)
        pred_df = pd.read_csv(pred_path)

        plot_loss_curve(city, history_df, plots_dir / f"loss_curve_{city}.png")
        plot_prediction(city, pred_df, args.horizon, args.tail_points, plots_dir / f"prediction_{city}_h{args.horizon}.png")
        print(f"saved city plots: {city}")

    plot_metrics_bar(metrics_df, plots_dir / "metric_bar_rmse_mae.png")
    print(f"saved metric bar: {plots_dir / 'metric_bar_rmse_mae.png'}")


if __name__ == "__main__":
    main()

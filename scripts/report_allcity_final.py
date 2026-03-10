import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final per-city plots and summary tables from merged predictions.")
    parser.add_argument(
        "--merged_csv",
        type=str,
        default="runs/gz_rain_allcities_final_predictions_merged.csv",
        help="Path to merged predictions csv.",
    )
    parser.add_argument(
        "--point_metrics",
        type=str,
        default="runs/gz_rain_allcities_point_bimamba_final/metrics_all.json",
        help="Path to all-city point metrics json.",
    )
    parser.add_argument(
        "--quantile_metrics",
        type=str,
        default="runs/gz_rain_allcities_quantile_bimamba_final/metrics_all.json",
        help="Path to all-city quantile metrics json.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/gz_rain_allcities_final_report",
        help="Output directory for final report assets.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon to report.")
    parser.add_argument("--tail_points", type=int, default=300, help="Tail points for each city plot.")
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


def load_metrics(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def maybe_get(dct, *keys):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_city_summary(point_df: pd.DataFrame, quant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in sorted(point_df["target_city"].tolist()):
        p = point_df[point_df["target_city"] == city].iloc[0].to_dict()
        q = quant_df[quant_df["target_city"] == city].iloc[0].to_dict()
        rows.append({
            "target_city": city,
            "point_rmse": p.get("rmse"),
            "point_mae": p.get("mae"),
            "point_bias_mean": p.get("bias_mean"),
            "point_ge10_mae": maybe_get(p, "rain_bands", "ge_10mm", "mae"),
            "point_ge25_mae": maybe_get(p, "rain_bands", "ge_25mm", "mae"),
            "point_ge50_mae": maybe_get(p, "rain_bands", "ge_50mm", "mae"),
            "quantile_rmse": q.get("rmse"),
            "quantile_mae": q.get("mae"),
            "quantile_picp": q.get("picp"),
            "quantile_pinaw": q.get("pinaw"),
            "quantile_aql": q.get("aql"),
            "quantile_ge10_mae": maybe_get(q, "rain_bands", "ge_10mm", "mae"),
            "quantile_ge25_mae": maybe_get(q, "rain_bands", "ge_25mm", "mae"),
            "quantile_ge50_mae": maybe_get(q, "rain_bands", "ge_50mm", "mae"),
            "quantile_ge10_picp": maybe_get(q, "rain_bands", "ge_10mm", "picp"),
            "quantile_ge25_picp": maybe_get(q, "rain_bands", "ge_25mm", "picp"),
            "quantile_ge50_picp": maybe_get(q, "rain_bands", "ge_50mm", "picp"),
            "point_better_mae": bool(p.get("mae") is not None and q.get("mae") is not None and p["mae"] <= q["mae"]),
        })
    return pd.DataFrame(rows)


def plot_city(df_city: pd.DataFrame, city: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df_city))

    ax.plot(x, df_city["y_true"].values, label="True", linewidth=1.8, color="black")
    ax.plot(x, df_city["point_pred"].values, label="Point Model", linewidth=1.5, color="tab:blue")
    ax.plot(
        x,
        df_city["interval_p50"].values,
        label="Interval Median",
        linewidth=1.4,
        color="tab:orange",
        linestyle="--",
    )
    ax.fill_between(
        x,
        df_city["y_lo"].values,
        df_city["y_hi"].values,
        alpha=0.2,
        color="tab:orange",
        label="Prediction Interval",
    )
    ax.set_title(f"{city} - Final Point and Interval Forecast")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Rainfall")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(summary_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cities = summary_df["target_city"].tolist()
    x = range(len(cities))
    width = 0.35

    axes[0].bar([i - width / 2 for i in x], summary_df["point_mae"], width=width, label="Point MAE")
    axes[0].bar([i + width / 2 for i in x], summary_df["quantile_mae"], width=width, label="Quantile MAE")
    axes[0].set_title("MAE by City")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(cities, rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], summary_df["quantile_picp"], width=width, label="PICP")
    axes[1].bar([i + width / 2 for i in x], summary_df["quantile_pinaw"], width=width, label="PINAW")
    axes[1].set_title("Interval Metrics by City")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(cities, rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_markdown(summary_df: pd.DataFrame) -> str:
    split_count = int(summary_df["point_better_mae"].sum())
    total = int(len(summary_df))
    lines = [
        "# 全省最终结果汇总",
        "",
        f"- 城市数: {total}",
        f"- 点预测模型在 MAE 上优于区间模型的城市数: {split_count}",
        "",
        "| 城市 | 点预测 MAE | 区间 MAE | PICP | PINAW | 点预测更优 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['target_city']} | {row['point_mae']:.4f} | {row['quantile_mae']:.4f} | "
            f"{row['quantile_picp']:.4f} | {row['quantile_pinaw']:.4f} | "
            f"{'是' if row['point_better_mae'] else '否'} |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    configure_matplotlib_for_chinese()

    merged_path = Path(args.merged_csv)
    point_metrics_path = Path(args.point_metrics)
    quant_metrics_path = Path(args.quantile_metrics)
    out_dir = Path(args.out_dir)
    city_plot_dir = out_dir / "city_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    city_plot_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.read_csv(merged_path)
    merged_df = merged_df[merged_df["horizon"] == int(args.horizon)].copy()
    point_df = load_metrics(point_metrics_path)
    quant_df = load_metrics(quant_metrics_path)
    summary_df = build_city_summary(point_df, quant_df).sort_values("target_city").reset_index(drop=True)

    for city in summary_df["target_city"]:
        city_df = merged_df[merged_df["target_city"] == city].copy()
        if "target_date" in city_df.columns:
            city_df = city_df.sort_values("target_date")
        if args.tail_points is not None and args.tail_points > 0:
            city_df = city_df.tail(int(args.tail_points))
        plot_city(city_df.reset_index(drop=True), city, city_plot_dir / f"{city}_final_report.png")

    plot_overview(summary_df, out_dir / "overview_metrics.png")
    summary_df.to_csv(out_dir / "final_city_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "final_city_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    with open(out_dir / "final_city_summary.md", "w", encoding="utf-8") as f:
        f.write(build_markdown(summary_df))

    print(f"saved summary csv: {out_dir / 'final_city_summary.csv'}")
    print(f"saved summary json: {out_dir / 'final_city_summary.json'}")
    print(f"saved summary md: {out_dir / 'final_city_summary.md'}")
    print(f"saved overview figure: {out_dir / 'overview_metrics.png'}")
    print(f"saved city plots to: {city_plot_dir}")


if __name__ == "__main__":
    main()

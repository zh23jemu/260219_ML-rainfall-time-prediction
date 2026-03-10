import argparse
import json
from pathlib import Path

import pandas as pd


POINT_EXPERIMENTS = {
    "贵阳市": "gz_rain_guiyang_point_bimamba_opt",
    "遵义市": "gz_rain_zunyi_point_bimamba_opt",
    "六盘水市": "gz_rain_liupanshui_point_bimamba_opt",
    "铜仁市": "gz_rain_tongren_point_bimamba_opt",
    "黔东南州": "gz_rain_qiandongnan_point_bimamba_opt",
}

QUANTILE_EXPERIMENTS = {
    "贵阳市": "gz_rain_guiyang_quantile_bimamba_asinh_v5",
    "遵义市": "gz_rain_zunyi_quantile_bimamba_asinh_v5",
    "六盘水市": "gz_rain_liupanshui_quantile_bimamba_asinh_v5",
    "铜仁市": "gz_rain_tongren_quantile_bimamba_asinh_v5",
    "黔东南州": "gz_rain_qiandongnan_quantile_bimamba_asinh_v5",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize minimal multi-city validation results.")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Root runs directory.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/gz_rain_multicity_minimal_summary",
        help="Directory to write summary csv/json/md.",
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_get(dct, *keys):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_rows(runs_dir: Path):
    rows = []
    for city, point_exp in POINT_EXPERIMENTS.items():
        point_metrics_path = runs_dir / point_exp / city / "metrics.json"
        quant_exp = QUANTILE_EXPERIMENTS[city]
        quant_metrics_path = runs_dir / quant_exp / city / "metrics.json"

        if not point_metrics_path.exists():
            raise FileNotFoundError(f"Missing point metrics: {point_metrics_path}")
        if not quant_metrics_path.exists():
            raise FileNotFoundError(f"Missing quantile metrics: {quant_metrics_path}")

        point_metrics = load_metrics(point_metrics_path)
        quant_metrics = load_metrics(quant_metrics_path)

        point_mae = point_metrics["mae"]
        quant_mae = quant_metrics["mae"]
        quant_picp = quant_metrics.get("picp")
        quant_pinaw = quant_metrics.get("pinaw")

        point_best = "point_model" if point_mae <= quant_mae else "quantile_model"
        interval_best = "quantile_model"
        split_optimal = point_best != interval_best

        rows.append({
            "target_city": city,
            "point_exp": point_exp,
            "quantile_exp": quant_exp,
            "point_rmse": point_metrics["rmse"],
            "point_mae": point_mae,
            "point_bias_mean": point_metrics.get("bias_mean"),
            "point_ge10_mae": maybe_get(point_metrics, "rain_bands", "ge_10mm", "mae"),
            "point_ge25_mae": maybe_get(point_metrics, "rain_bands", "ge_25mm", "mae"),
            "point_ge50_mae": maybe_get(point_metrics, "rain_bands", "ge_50mm", "mae"),
            "quantile_rmse": quant_metrics["rmse"],
            "quantile_mae": quant_mae,
            "quantile_picp": quant_picp,
            "quantile_pinaw": quant_pinaw,
            "quantile_aql": quant_metrics.get("aql"),
            "quantile_ge10_mae": maybe_get(quant_metrics, "rain_bands", "ge_10mm", "mae"),
            "quantile_ge25_mae": maybe_get(quant_metrics, "rain_bands", "ge_25mm", "mae"),
            "quantile_ge50_mae": maybe_get(quant_metrics, "rain_bands", "ge_50mm", "mae"),
            "quantile_ge10_picp": maybe_get(quant_metrics, "rain_bands", "ge_10mm", "picp"),
            "quantile_ge25_picp": maybe_get(quant_metrics, "rain_bands", "ge_25mm", "picp"),
            "quantile_ge50_picp": maybe_get(quant_metrics, "rain_bands", "ge_50mm", "picp"),
            "point_best_by_mae": point_best,
            "interval_best": interval_best,
            "split_optimal": split_optimal,
        })
    return rows


def build_summary_payload(df: pd.DataFrame):
    split_count = int(df["split_optimal"].sum())
    total = int(len(df))
    return {
        "total_cities": total,
        "split_optimal_count": split_count,
        "split_optimal_ratio": float(split_count / total) if total else 0.0,
        "cities": df["target_city"].tolist(),
        "point_model_better_mae_cities": df.loc[df["point_best_by_mae"] == "point_model", "target_city"].tolist(),
        "quantile_model_better_mae_cities": df.loc[df["point_best_by_mae"] == "quantile_model", "target_city"].tolist(),
    }


def build_markdown(df: pd.DataFrame, summary: dict):
    lines = [
        "# 贵州多城市最小验证汇总",
        "",
        f"- 城市数: {summary['total_cities']}",
        f"- 出现“点预测模型/区间模型分开最优”的城市数: {summary['split_optimal_count']}",
        f"- 占比: {summary['split_optimal_ratio']:.2%}",
        "",
        "| 城市 | 点预测最优模型 | 点预测MAE | 区间模型MAE | PICP | PINAW | 是否分开最优 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['target_city']} | {row['point_best_by_mae']} | "
            f"{row['point_mae']:.4f} | {row['quantile_mae']:.4f} | "
            f"{row['quantile_picp']:.4f} | {row['quantile_pinaw']:.4f} | "
            f"{'是' if row['split_optimal'] else '否'} |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(build_rows(runs_dir)).sort_values("target_city").reset_index(drop=True)
    summary = build_summary_payload(df)

    csv_path = out_dir / "multicity_minimal_summary.csv"
    json_path = out_dir / "multicity_minimal_summary.json"
    md_path = out_dir / "multicity_minimal_summary.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(df, summary))

    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved md: {md_path}")


if __name__ == "__main__":
    main()

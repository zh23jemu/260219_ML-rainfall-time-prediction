import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge all-city point and interval prediction results.")
    parser.add_argument(
        "--point_csv",
        type=str,
        default="runs/gz_rain_allcities_point_bimamba_final/predictions_all.csv",
        help="Path to all-city point prediction csv.",
    )
    parser.add_argument(
        "--quantile_csv",
        type=str,
        default="runs/gz_rain_allcities_quantile_bimamba_final/predictions_all.csv",
        help="Path to all-city quantile prediction csv.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="runs/gz_rain_allcities_final_predictions_merged.csv",
        help="Output merged csv path.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame, expected_cols):
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.copy()


def main():
    args = parse_args()

    point_path = Path(args.point_csv)
    quantile_path = Path(args.quantile_csv)
    out_path = Path(args.out_csv)

    if not point_path.exists():
        raise FileNotFoundError(f"Point csv not found: {point_path}")
    if not quantile_path.exists():
        raise FileNotFoundError(f"Quantile csv not found: {quantile_path}")

    point_df = pd.read_csv(point_path)
    quantile_df = pd.read_csv(quantile_path)

    key_cols = ["target_city", "target_date", "horizon"]
    point_df = normalize_columns(point_df, key_cols + ["y_true", "y_pred"])
    quantile_df = normalize_columns(quantile_df, key_cols + ["y_true", "y_pred", "y_lo", "y_hi"])

    point_df = point_df.rename(columns={"y_pred": "point_pred", "y_true": "y_true_point"})
    quantile_df = quantile_df.rename(
        columns={
            "y_pred": "interval_p50",
            "y_true": "y_true_quantile",
        }
    )

    merged = pd.merge(
        point_df[key_cols + ["sample", "y_true_point", "point_pred"]],
        quantile_df[key_cols + ["sample", "y_true_quantile", "interval_p50", "y_lo", "y_hi"]],
        on=key_cols,
        how="inner",
        suffixes=("_point", "_quantile"),
    )

    merged["y_true"] = merged["y_true_point"]
    mismatch = (merged["y_true_point"] - merged["y_true_quantile"]).abs() > 1e-6
    if mismatch.any():
        raise ValueError("Found mismatched y_true values between point and quantile predictions.")

    merged = merged[
        [
            "target_city",
            "target_date",
            "horizon",
            "sample_point",
            "sample_quantile",
            "y_true",
            "point_pred",
            "interval_p50",
            "y_lo",
            "y_hi",
        ]
    ].rename(columns={"sample_point": "sample_point_model", "sample_quantile": "sample_interval_model"})

    merged = merged.sort_values(["target_city", "target_date", "horizon"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved merged predictions: {out_path}")
    print(f"rows: {len(merged)}")


if __name__ == "__main__":
    main()

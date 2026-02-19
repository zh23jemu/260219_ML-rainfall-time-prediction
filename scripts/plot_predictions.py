import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot rainfall predictions by city.")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to predictions_all.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for images")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon to plot (1-based)")
    parser.add_argument("--tail_points", type=int, default=300, help="Tail points for each city plot")
    parser.add_argument("--max_cities", type=int, default=None, help="Optional max number of cities to plot")
    return parser.parse_args()


def plot_one_city(df_city: pd.DataFrame, city: str, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df_city))
    ax.plot(x, df_city["y_true"].values, label="True", linewidth=1.6)
    ax.plot(x, df_city["y_pred"].values, label="Pred", linewidth=1.4)

    if "y_lo" in df_city.columns and "y_hi" in df_city.columns:
        ax.fill_between(
            x,
            df_city["y_lo"].values,
            df_city["y_hi"].values,
            alpha=0.2,
            label="PI",
        )

    ax.set_title(f"{city} - True vs Pred")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Rainfall")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    horizon_df = df[df["horizon"] == int(args.horizon)].copy()
    cities = sorted(horizon_df["target_city"].unique())
    if args.max_cities is not None:
        cities = cities[: args.max_cities]

    for city in cities:
        cdf = horizon_df[horizon_df["target_city"] == city].copy()
        if args.tail_points is not None and args.tail_points > 0:
            cdf = cdf.tail(int(args.tail_points))
        out_name = f"{city}_h{args.horizon}_timeseries.png"
        out_path = os.path.join(args.out_dir, out_name)
        plot_one_city(cdf.reset_index(drop=True), city, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

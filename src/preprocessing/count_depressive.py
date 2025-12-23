import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]


def plot_distribution(csv_path: Path, column_name: str, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_path}")

    counts = df[column_name].value_counts()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    hist_path = output_dir / f"{csv_path.stem}_{column_name}_hist.png"
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, color="royalblue")
    plt.xlabel(column_name, fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title(f"Frequency of {column_name}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    sns.despine()
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    box_path = output_dir / f"{csv_path.stem}_{column_name}_box.png"
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=counts.values, color="darkorange")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title(f"Boxplot of {column_name} counts", fontsize=14, fontweight="bold")
    sns.despine()
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram to {hist_path}")
    print(f"Saved boxplot to {box_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class distribution for the depressive label.")
    parser.add_argument(
        "--arquivo",
        type=Path,
        default=ROOT / "data/raw/dataset_final.csv",
        help="CSV containing the label column.",
    )
    parser.add_argument("--coluna", default="depressive", help="Column name to plot.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "assets/figures",
        help="Directory to store the generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_distribution(args.arquivo, args.coluna, args.saida)


if __name__ == "__main__":
    main()

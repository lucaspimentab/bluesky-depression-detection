import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def split_csv(file_path: Path, output_dir: Path, chunk_size: int = 500) -> None:
    """Split a CSV into smaller parts of ``chunk_size`` rows each."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = file_path.stem

    for idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size), start=1):
        output_file = output_dir / f"{file_name}_part{idx}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a large CSV into smaller chunks.")
    parser.add_argument(
        "--arquivo",
        type=Path,
        default=ROOT / "data/raw/dataset_2 - merged.csv",
        help="CSV to be split.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "data/derived/chunks",
        help="Directory to save the parts.",
    )
    parser.add_argument("--tamanho", type=int, default=500, help="Rows per chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_csv(args.arquivo, args.saida, args.tamanho)


if __name__ == "__main__":
    main()

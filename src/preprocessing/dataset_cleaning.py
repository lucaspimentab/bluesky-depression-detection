import argparse
from pathlib import Path

import emoji
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def remove_emojis(text: str) -> str:
    """Strip emoji characters from a string, keeping non-string inputs untouched."""
    return emoji.replace_emoji(text, replace="") if isinstance(text, str) else text


def clean_datasets(
    dataset1: Path,
    dataset2: Path,
    output_dataset: Path,
    characterization_output: Path,
    counts_output: Path,
) -> None:
    """Combine the two raw sources into a single cleaned dataset."""
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)

    # Keep only positive rows from dataset 2 and drop duplicates.
    df2 = df2.drop_duplicates(subset=["text"])
    df2 = df2.loc[df2["depressive"] == 1]
    df2["depressive"] = df2["depressive"].astype("Int64")

    # Remove unused metadata columns from dataset 1 and sanitize labels.
    drop_cols = ["repostCount", "replyCount", "link", "image", "createdAt"]
    df1 = df1.drop(columns=[c for c in drop_cols if c in df1.columns])
    df1["depressive"] = df1["depressive"].astype(str).str.strip()
    df1 = df1[df1["depressive"].isin(["0", "1"])]
    df1["depressive"] = df1["depressive"].astype("Int64")

    characterization_output.parent.mkdir(parents=True, exist_ok=True)
    df1.to_csv(characterization_output, index=False)

    # Merge, keep only the relevant columns, and strip emojis.
    combined = pd.concat([df1, df2], axis=0)
    combined = combined[["text", "depressive"]]
    combined["text"] = combined["text"].apply(remove_emojis)
    combined = combined[combined["text"].astype(str).str.strip() != ""]
    combined["depressive"] = combined["depressive"].astype("Int64")

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dataset, index=False)

    counts_output.parent.mkdir(parents=True, exist_ok=True)
    combined["depressive"].value_counts().to_csv(counts_output, index=True)

    print(f"Clean dataset saved to {output_dataset}")
    print(f"Class counts saved to {counts_output}")
    print(f"Metadata characterization saved to {characterization_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and merge the two raw Bluesky datasets.")
    parser.add_argument(
        "--dataset1",
        type=Path,
        default=ROOT / "data/raw/dataset_1_final.csv",
        help="Primary CSV with text and metadata.",
    )
    parser.add_argument(
        "--dataset2",
        type=Path,
        default=ROOT / "data/raw/dataset_2_final.csv",
        help="Secondary CSV (only positives are kept).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "data/raw/dataset_final.csv",
        help="Destination for the consolidated dataset.",
    )
    parser.add_argument(
        "--caracterizacao",
        type=Path,
        default=ROOT / "data/derived/dataset-characterization.csv",
        help="Destination for the cleaned metadata-only table.",
    )
    parser.add_argument(
        "--contagem",
        type=Path,
        default=ROOT / "data/derived/depressive_count.csv",
        help="Destination for label counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_datasets(
        dataset1=args.dataset1,
        dataset2=args.dataset2,
        output_dataset=args.saida,
        characterization_output=args.caracterizacao,
        counts_output=args.contagem,
    )


if __name__ == "__main__":
    main()

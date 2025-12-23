import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from predict_text import DepressionClassifier

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliacao rapida do classificador BERT treinado.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/raw/dataset_final_3003.csv",
        help="CSV com colunas 'text' e 'depressive'.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "data/derived",
        help="Diretorio para salvar falsos positivos/negativos.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para o split de validacao.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df["text"] = df["text"].astype(str).fillna("")

    texts = df["text"].tolist()
    labels = df["depressive"].tolist()

    _, val_texts, _, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=args.seed)

    classifier = DepressionClassifier()

    false_positive_count = classifier.count_false_positives(val_texts, val_labels)
    print(f"Numero de falsos positivos: {false_positive_count}")

    false_negative_count = classifier.count_false_negatives(val_texts, val_labels)
    print(f"Numero de falsos negativos: {false_negative_count}")

    false_positives = classifier.get_false_positives(val_texts, val_labels)
    false_negatives = classifier.get_false_negatives(val_texts, val_labels)

    fp_path = args.saida / "false_positives_dataset.csv"
    fn_path = args.saida / "false_negatives_dataset.csv"
    pd.DataFrame(false_positives).to_csv(fp_path, index=False)
    pd.DataFrame(false_negatives).to_csv(fn_path, index=False)
    print(f"Falsos positivos salvos em: {fp_path}")
    print(f"Falsos negativos salvos em: {fn_path}")


if __name__ == "__main__":
    main()

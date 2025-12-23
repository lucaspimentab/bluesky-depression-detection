import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]


def gerar_divisoes(
    arquivo: Path,
    coluna_texto: str,
    coluna_alvo: str,
    saida: Path,
    tamanho_validacao: float,
    tamanho_teste: float,
    seed: int,
) -> None:
    saida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(arquivo)
    df[coluna_texto] = df[coluna_texto].astype(str).fillna("")
    df = df[df[coluna_texto].str.strip() != ""]
    df[coluna_alvo] = pd.to_numeric(df[coluna_alvo], errors="coerce")
    df = df.dropna(subset=[coluna_alvo])
    df[coluna_alvo] = df[coluna_alvo].astype(int)

    splitter_teste = StratifiedShuffleSplit(
        n_splits=1,
        test_size=tamanho_teste,
        random_state=seed,
    )
    idx_treino, idx_teste = next(splitter_teste.split(df, df[coluna_alvo]))
    df_treino = df.iloc[idx_treino].reset_index(drop=True)
    df_teste = df.iloc[idx_teste].reset_index(drop=True)

    proporcao_validacao = tamanho_validacao / (1 - tamanho_teste)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=proporcao_validacao,
        random_state=seed,
    )
    idx_treino2, idx_val = next(splitter_val.split(df_treino, df_treino[coluna_alvo]))
    df_final_treino = df_treino.iloc[idx_treino2].reset_index(drop=True)
    df_validacao = df_treino.iloc[idx_val].reset_index(drop=True)

    df_final_treino.to_csv(saida / "treino.csv", index=False)
    df_validacao.to_csv(saida / "validacao.csv", index=False)
    df_teste.to_csv(saida / "teste.csv", index=False)

    resumo = {
        "treino": df_final_treino[coluna_alvo].value_counts().to_dict(),
        "validacao": df_validacao[coluna_alvo].value_counts().to_dict(),
        "teste": df_teste[coluna_alvo].value_counts().to_dict(),
    }
    (saida / "resumo_classes.json").write_text(pd.Series(resumo).to_json(indent=2), encoding="utf-8")
    print(f"Splits saved to {saida.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Bluesky dataset into train/validation/test with stratification."
    )
    parser.add_argument(
        "--arquivo",
        type=Path,
        default=ROOT / "data/raw/dataset_final.csv",
        help="CSV with texts, metadata, and the target column.",
    )
    parser.add_argument("--coluna-texto", default="text")
    parser.add_argument("--coluna-alvo", default="depressive")
    parser.add_argument(
        "--saida",
        type=Path,
        default=ROOT / "data/processed/splits/default",
        help="Output directory for the split files.",
    )
    parser.add_argument("--validacao", type=float, default=0.2, help="Validation proportion.")
    parser.add_argument("--teste", type=float, default=0.2, help="Test proportion.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gerar_divisoes(
        arquivo=args.arquivo,
        coluna_texto=args.coluna_texto,
        coluna_alvo=args.coluna_alvo,
        saida=args.saida,
        tamanho_validacao=args.validacao,
        tamanho_teste=args.teste,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

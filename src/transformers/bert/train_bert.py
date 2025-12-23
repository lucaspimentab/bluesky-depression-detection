import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[3]


class TextDataset(Dataset):
    """Dataset simples que converte textos em tensores usando o tokenizer informado."""

    def __init__(self, textos: List[str], rotulos: List[int], tokenizer: BertTokenizer, max_length: int = 128):
        self.textos = textos
        self.rotulos = rotulos
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, indice: int):
        encoding = self.tokenizer(
            self.textos[indice],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.rotulos[indice], dtype=torch.long),
        }


def carregar_dados(
    arquivo: Optional[Path],
    diretorio_splits: Optional[Path],
    coluna_texto: str,
    coluna_alvo: str,
    tamanho_validacao: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados a partir de um CSV único ou de um diretório com treino/validação."""

    if diretorio_splits:
        treino = pd.read_csv(diretorio_splits / "treino.csv")
        validacao = pd.read_csv(diretorio_splits / "validacao.csv")
    else:
        if arquivo is None:
            raise ValueError("Informe um CSV com --arquivo ou um diretório de splits com --diretorio-splits.")
        base = pd.read_csv(arquivo)
        base[coluna_texto] = base[coluna_texto].astype(str).fillna("")
        base = base[base[coluna_texto].str.strip() != ""]
        base = base.drop_duplicates(subset=[coluna_texto])
        base[coluna_alvo] = pd.to_numeric(base[coluna_alvo], errors="coerce")
        base = base.dropna(subset=[coluna_alvo])
        base[coluna_alvo] = base[coluna_alvo].astype(int)

        treino, validacao = train_test_split(
            base,
            test_size=tamanho_validacao,
            random_state=seed,
            stratify=base[coluna_alvo],
        )

    return treino.reset_index(drop=True), validacao.reset_index(drop=True)


def preparar_listas(
    df: pd.DataFrame,
    coluna_texto: str,
    coluna_alvo: str,
    aplicar_oversampling: bool,
    seed: int,
) -> Tuple[List[str], List[int]]:
    """Extrai listas de textos e rótulos com opção de oversampling."""

    textos = df[coluna_texto].astype(str).tolist()
    rotulos = df[coluna_alvo].astype(int).tolist()

    if aplicar_oversampling:
        ros = RandomOverSampler(random_state=seed)
        textos, rotulos = ros.fit_resample(pd.DataFrame(textos), pd.DataFrame(rotulos))
        textos = textos[0].tolist()
        rotulos = rotulos[0].tolist()

    return textos, rotulos


def calcular_metricas(modelo, dataloader, salvar_probabilidades: Optional[Path] = None) -> None:
    modelo.eval()
    todas_labels = []
    todas_predicoes = []
    todas_probs = []

    with torch.no_grad():
        for lote in tqdm(dataloader, desc="Avaliação", leave=False):
            input_ids = lote["input_ids"].to(device)
            attention_mask = lote["attention_mask"].to(device)
            labels = lote["labels"].to(device)

            saida = modelo(input_ids, attention_mask=attention_mask)
            logits = saida.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            predicoes = torch.argmax(logits, dim=1)

            todas_labels.extend(labels.cpu().tolist())
            todas_predicoes.extend(predicoes.cpu().tolist())
            todas_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(todas_labels, todas_predicoes)
    precision = precision_score(todas_labels, todas_predicoes, zero_division=0)
    recall = recall_score(todas_labels, todas_predicoes, zero_division=0)
    f1 = f1_score(todas_labels, todas_predicoes, zero_division=0)
    matriz = confusion_matrix(todas_labels, todas_predicoes)

    print("\n=== Métricas de validação ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Matriz de confusão:\n{matriz}")

    if salvar_probabilidades:
        salvar_probabilidades.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prob": todas_probs, "rotulo": todas_labels}).to_csv(salvar_probabilidades, index=False)


def treinar_bert(
    df_treino: pd.DataFrame,
    df_validacao: pd.DataFrame,
    coluna_texto: str,
    coluna_alvo: str,
    modelo_saida: Path,
    prob_saida: Optional[Path],
    epochs: int,
    batch_size: int,
    max_length: int,
    oversampling: bool,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    dropout: Optional[float],
) -> None:
    textos_treino, rotulos_treino = preparar_listas(df_treino, coluna_texto, coluna_alvo, oversampling, seed)
    textos_validacao = df_validacao[coluna_texto].astype(str).tolist()
    rotulos_validacao = df_validacao[coluna_alvo].astype(int).tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_treino = TextDataset(textos_treino, rotulos_treino, tokenizer, max_length=max_length)
    dataset_validacao = TextDataset(textos_validacao, rotulos_validacao, tokenizer, max_length=max_length)

    loader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
    loader_validacao = DataLoader(dataset_validacao, batch_size=batch_size, shuffle=False)

    modelo = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    if dropout is not None:
        modelo.config.attention_probs_dropout_prob = dropout
        modelo.config.hidden_dropout_prob = dropout

    modelo.to(device)

    otimizador = AdamW(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(loader_treino) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        otimizador,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoca in range(epochs):
        modelo.train()
        perdas = []
        barra = tqdm(loader_treino, desc=f"Treinando época {epoca+1}", leave=False)
        for lote in barra:
            input_ids = lote["input_ids"].to(device)
            attention_mask = lote["attention_mask"].to(device)
            labels = lote["labels"].to(device)

            otimizador.zero_grad()
            saida = modelo(input_ids, attention_mask=attention_mask, labels=labels)
            perda = saida.loss
            perda.backward()
            otimizador.step()
            scheduler.step()

            perdas.append(perda.item())
            barra.set_postfix(loss=perda.item())

        media_perda = sum(perdas) / max(1, len(perdas))
        print(f"Época {epoca+1}: perda média {media_perda:.4f}")

    calcular_metricas(modelo, loader_validacao, prob_saida)

    modelo_saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(modelo_saida)
    tokenizer.save_pretrained(modelo_saida)
    print(f"Modelo e tokenizer salvos em {modelo_saida.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento de BERT para detecção de depressão em postagens do Bluesky."
    )
    parser.add_argument("--arquivo", type=Path, help="CSV completo com textos e rótulos.")
    parser.add_argument(
        "--diretorio-splits",
        type=Path,
        help="Diretório com arquivos treino.csv e validacao.csv. Tem precedência sobre --arquivo.",
    )
    parser.add_argument("--coluna-texto", default="text", help="Nome da coluna de texto.")
    parser.add_argument("--coluna-alvo", default="depressive", help="Nome da coluna alvo (0/1).")
    parser.add_argument("--tamanho-validacao", type=float, default=0.2, help="Proporção para validação se usar CSV único.")
    parser.add_argument("--sem-oversampling", action="store_true", help="Desativa o RandomOverSampler.")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas de treinamento.")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do lote.")
    parser.add_argument("--max-length", type=int, default=128, help="Comprimento máximo dos textos tokenizados.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Taxa de aprendizado do AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2) do AdamW.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Proporção de passos de warmup (0-1).")
    parser.add_argument("--dropout", type=float, help="Valor opcional para dropout (sobrescreve o padrão do modelo).")
    parser.add_argument("--saida-modelo", type=Path, default=Path("modelos/bert"), help="Diretório de saída do modelo.")
    parser.add_argument(
        "--saida-probabilidades",
        type=Path,
        default=None,
        help="CSV opcional para salvar probabilidades da validação.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df_treino, df_validacao = carregar_dados(
        arquivo=args.arquivo,
        diretorio_splits=args.diretorio_splits,
        coluna_texto=args.coluna_texto,
        coluna_alvo=args.coluna_alvo,
        tamanho_validacao=args.tamanho_validacao,
        seed=args.seed,
    )

    treinar_bert(
        df_treino=df_treino,
        df_validacao=df_validacao,
        coluna_texto=args.coluna_texto,
        coluna_alvo=args.coluna_alvo,
        modelo_saida=Path(args.saida_modelo),
        prob_saida=Path(args.saida_probabilidades) if args.saida_probabilidades else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        oversampling=not args.sem_oversampling,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()

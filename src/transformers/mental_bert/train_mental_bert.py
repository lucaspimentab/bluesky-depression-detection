import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, textos: List[str], rotulos: List[int], tokenizer: BertTokenizer, max_length: int = 128):
        self.textos = textos
        self.rotulos = rotulos
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, indice: int):
        codificacao = self.tokenizer(
            self.textos[indice],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": codificacao["input_ids"].squeeze(0),
            "attention_mask": codificacao["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.rotulos[indice], dtype=torch.long),
        }


def carregar_dados(
    arquivo: Optional[Path],
    diretorio: Optional[Path],
    coluna_texto: str,
    coluna_alvo: str,
    tamanho_validacao: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if diretorio:
        treino = pd.read_csv(diretorio / "treino.csv")
        validacao = pd.read_csv(diretorio / "validacao.csv")
    else:
        if arquivo is None:
            raise ValueError("Informe um CSV com --arquivo ou utilize --diretorio-splits.")
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
            stratify=base[coluna_alvo],
            random_state=seed,
        )

    return treino.reset_index(drop=True), validacao.reset_index(drop=True)


def aplicar_oversampling(textos: List[str], rotulos: List[int], seed: int) -> Tuple[List[str], List[int]]:
    ros = RandomOverSampler(random_state=seed)
    textos_res, rotulos_res = ros.fit_resample(pd.DataFrame(textos), pd.DataFrame(rotulos))
    return textos_res[0].tolist(), rotulos_res[0].tolist()


def avaliar(modelo, dataloader, caminho_prob: Optional[Path]) -> None:
    modelo.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for lote in tqdm(dataloader, desc="Avaliação", leave=False):
            ids = lote["input_ids"].to(device)
            mask = lote["attention_mask"].to(device)
            y = lote["labels"].to(device)

            saida = modelo(ids, attention_mask=mask)
            logits = saida.logits
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            labels.extend(y.cpu().tolist())
            preds.extend(pred.cpu().tolist())
            probs.extend(prob.cpu().tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    matriz = confusion_matrix(labels, preds)

    print("\n=== Métricas de validação ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Matriz de confusão:\n{matriz}")

    if caminho_prob:
        caminho_prob.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prob": probs, "rotulo": labels}).to_csv(caminho_prob, index=False)


def treinar(
    df_treino: pd.DataFrame,
    df_validacao: pd.DataFrame,
    coluna_texto: str,
    coluna_alvo: str,
    oversampling: bool,
    epochs: int,
    batch_size: int,
    max_length: int,
    seed: int,
    saida_modelo: Path,
    saida_prob: Optional[Path],
    modelo_pre_treinado: str,
) -> None:
    textos_treino = df_treino[coluna_texto].astype(str).tolist()
    rotulos_treino = df_treino[coluna_alvo].astype(int).tolist()

    if oversampling:
        textos_treino, rotulos_treino = aplicar_oversampling(textos_treino, rotulos_treino, seed)

    textos_val = df_validacao[coluna_texto].astype(str).tolist()
    rotulos_val = df_validacao[coluna_alvo].astype(int).tolist()

    tokenizer = BertTokenizer.from_pretrained(modelo_pre_treinado)
    dataset_treino = TextDataset(textos_treino, rotulos_treino, tokenizer, max_length=max_length)
    dataset_val = TextDataset(textos_val, rotulos_val, tokenizer, max_length=max_length)

    loader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    modelo = BertForSequenceClassification.from_pretrained(modelo_pre_treinado, num_labels=2)
    modelo.to(device)
    otimizador = AdamW(modelo.parameters(), lr=2e-5)

    for epoca in range(epochs):
        modelo.train()
        perdas = []
        barra = tqdm(loader_treino, desc=f"Treinando época {epoca+1}", leave=False)
        for lote in barra:
            ids = lote["input_ids"].to(device)
            mask = lote["attention_mask"].to(device)
            y = lote["labels"].to(device)

            otimizador.zero_grad()
            saida = modelo(ids, attention_mask=mask, labels=y)
            perda = saida.loss
            perda.backward()
            otimizador.step()

            perdas.append(perda.item())
            barra.set_postfix(loss=perda.item())

        media = sum(perdas) / max(len(perdas), 1)
        print(f"Época {epoca+1}: perda média {media:.4f}")

    avaliar(modelo, loader_val, saida_prob)

    saida_modelo.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(saida_modelo)
    tokenizer.save_pretrained(saida_modelo)
    print(f"Modelo salvo em {saida_modelo.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning do MentalBERT para o dataset Bluesky.")
    parser.add_argument("--arquivo", type=Path, help="CSV único com textos e rótulos.")
    parser.add_argument("--diretorio-splits", type=Path, help="Diretório com treino.csv/validacao.csv.")
    parser.add_argument("--coluna-texto", default="text")
    parser.add_argument("--coluna-alvo", default="depressive")
    parser.add_argument("--tamanho-validacao", type=float, default=0.2)
    parser.add_argument("--sem-oversampling", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modelo-pre-treinado",
        default="mental/mental-bert-base-uncased",
        help="Identificador do modelo pré-treinado no HuggingFace.",
    )
    parser.add_argument("--saida-modelo", type=Path, default=Path("modelos/mental_bert"))
    parser.add_argument("--saida-probabilidades", type=Path, help="CSV para salvar probabilidades da validação.")
    return parser.parse_args()


def main():
    args = parse_args()
    df_treino, df_val = carregar_dados(
        arquivo=args.arquivo,
        diretorio=args.diretorio_splits,
        coluna_texto=args.coluna_texto,
        coluna_alvo=args.coluna_alvo,
        tamanho_validacao=args.tamanho_validacao,
        seed=args.seed,
    )

    treinar(
        df_treino=df_treino,
        df_validacao=df_val,
        coluna_texto=args.coluna_texto,
        coluna_alvo=args.coluna_alvo,
        oversampling=not args.sem_oversampling,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        saida_modelo=args.saida_modelo,
        saida_prob=args.saida_probabilidades,
        modelo_pre_treinado=args.modelo_pre_treinado,
    )


if __name__ == "__main__":
    main()

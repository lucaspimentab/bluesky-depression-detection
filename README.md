# Bluesky-Depression

Classificação de conteúdo depressivo em posts do Bluesky.

## Estrutura
- `data/raw`: CSVs originais/limpos.
- `data/derived`: artefatos auxiliares (contagens, probabilidades de validação, falsos positivos/negativos).
- `data/processed/splits/default`: splits estratificados (treino/validação/teste).
- `src/preprocessing`: limpeza, divisão e utilitários de dataset.
- `src/transformers`: fine-tuning e avaliação rápida dos modelos de texto.
- `notebooks`: análises exploratórias.
- `assets/figures`: figuras geradas (word clouds, histogramas, etc.).
- `docs/papers`: paper

## Requisitos
Python 3.12+ com pandas, scikit-learn, seaborn, matplotlib, emoji, PyTorch e Hugging Face Transformers instalados.

## Fluxo básico
```bash
# 1) Limpar e mesclar fontes brutas em um único dataset e contagem de classes
python src/preprocessing/dataset_cleaning.py

# 2) Criar splits estratificados (treino/val/teste)
python src/preprocessing/split_dataset.py

# 3) Treinar BERT (exemplo) e salvar modelo/tokenizer
python src/transformers/bert/train_bert.py \
  --arquivo data/raw/dataset_final.csv \
  --saida-modelo outputs/models/bert

# 4) Visualizar distribuição da coluna depressive
python src/preprocessing/count_depressive.py \
  --arquivo data/raw/dataset_final.csv \
  --saida assets/figures

# 5) Avaliação rápida de falsos positivos/negativos (após treinar o modelo)
python src/transformers/bert/main.py \
  --dataset data/raw/dataset_final_3003.csv \
  --saida data/derived
```
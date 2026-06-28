import os
import glob
import pandas as pd
import kagglehub
from src.dependencias import Dependencia

_dep = Dependencia


def leitura() -> pd.DataFrame:
    """
    Baixa o dataset do ENEM 2023 via Kaggle, filtra por MG,
    seleciona as colunas relevantes e retorna uma amostra de n_amostra linhas.

    Na primeira execução, baixa via Kaggle e salva CSV local.
    Nas execuções seguintes, lê direto do CSV local (mais rápido).
    """
    print("Leitura dos Arquivos...")

    cols = [
        "SG_UF_PROVA",
        "TP_FAIXA_ETARIA",
        "TP_SEXO", "TP_COR_RACA",
        "TP_ESCOLA",
        "IN_TREINEIRO",
        "Q001", "Q002", "Q006", "Q025",
        "NU_NOTA_REDACAO",
    ]

    if not os.path.isfile(_dep.path_dataset):
        print("Baixando dataset do Kaggle...")
        path = kagglehub.dataset_download(_dep.kaggle_dataset)
        print(f"  Dataset baixado em: {path}")

        # Localiza o CSV dentro da pasta baixada
        csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not csvs:
            raise FileNotFoundError(f"Nenhum CSV encontrado em: {path}")

        print(f"  Arquivos encontrados: {[os.path.basename(c) for c in csvs]}")
        csv_path = csvs[0]  # usa o primeiro (normalmente há só um)

        os.makedirs(os.path.dirname(_dep.path_dataset), exist_ok=True)
        df = pd.read_csv(csv_path, usecols=cols, sep=",", encoding="utf-8")
        df.to_csv(_dep.path_dataset, index=False)
        print(f"  Dataset salvo em '{_dep.path_dataset}' para uso futuro.")
    else:
        print(f"  Carregando dataset local: '{_dep.path_dataset}'...")
        df = pd.read_csv(_dep.path_dataset, usecols=cols)

    print(f"  Total de registros: {len(df)}")

    # Filtra somente MG
    df = df[df["SG_UF_PROVA"] == "MG"].drop(columns=["SG_UF_PROVA"])
    print(f"  Registros MG: {len(df)}")

    # Amostragem
    n = min(_dep.n_amostra, len(df))
    if n < _dep.n_amostra:
        print(f"  [AVISO] Apenas {n} registros disponíveis; usando todos.")
    df_amostra = df.sample(n=n, random_state=42).reset_index(drop=True)

    caminho_saida = f"data/enem_2023_mg_amostra_{n}.csv"
    df_amostra.to_csv(caminho_saida, index=False)

    print(f"Leitura concluída: {len(df_amostra)} registros salvos em '{caminho_saida}'...")
    return df_amostra
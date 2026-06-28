import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def pre_processamento(df: pd.DataFrame):
    """
    Executa o pré-processamento completo:
      1. Remove alunos ausentes/eliminados (nota redação NaN).
      2. Cria variável alvo binária: 1 se nota >= 600, 0 caso contrário.
      3. Codifica variáveis binárias:
           TP_SEXO      : M → 1 / F → 0
           IN_TREINEIRO : já é 0/1
           Q025         : A (Não) → 0 / B–E (Sim) → 1
      4. Aplica One-Hot Encoding nas demais categóricas:
           TP_FAIXA_ETARIA, TP_COR_RACA, TP_ESCOLA, Q001, Q002, Q006
      5. Normalização MinMax em todas as colunas de atributos.
      6. Separa X (atributos) e y (alvo).

    Retorna: X, y, df_processado
    """
    print("Pré-processamento...")

    df = df.copy()

    # Remover linhas sem nota de redação (faltou ou foi eliminado)
    antes = len(df)
    df = df.dropna(subset=["NU_NOTA_REDACAO"])
    print(f"  Registros removidos (sem nota): {antes - len(df)}")

    # Variável alvo binária
    #    Classe 1 → nota >= 600  |  Classe 0 → nota < 600
    df["ALVO"] = (df["NU_NOTA_REDACAO"] >= 600).astype(int)
    df = df.drop(columns=["NU_NOTA_REDACAO"])

    # Codificação de variáveis binárias
    #    TP_SEXO: M → 1 / F → 0
    df["TP_SEXO"] = df["TP_SEXO"].map({"M": 1, "F": 0})

    # Q025: acesso à internet em casa
    #    A =  0  |  B, C, D, E = 1
    df["Q025"] = df["Q025"].map(lambda v: 0 if v == "A" else (1 if pd.notna(v) else None))

    # One-Hot Encoding nas variáveis categóricas
    colunas_categoricas = ["TP_FAIXA_ETARIA", "TP_COR_RACA", "TP_ESCOLA", "Q001", "Q002", "Q006"]
    df_processado = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

    # Separar atributos (X) e alvo (y)
    X = df_processado.drop(columns=["ALVO"])
    y = df_processado["ALVO"]

    # Normalização MinMax [0, 1]
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    print(f"  Atributos após encoding e normalização: {X_scaled.shape[1]} |  Amostras: {X_scaled.shape[0]}")
    print("Pré-processamento concluído.")
    return X_scaled, y, df_processado
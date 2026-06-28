import pandas as pd
from sklearn.model_selection import train_test_split


def dividir(X: pd.DataFrame, y: pd.Series):
    """
    Divide os dados em Treino (60%), Validação (20%) e Teste (20%)
    com estratificação pelo alvo.

    Retorna: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("Divisão dos dados...")

    # 20% teste
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 25% de 80% = 20% total para validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
    )

    print(f"Treino:     {X_train.shape[0]} amostras  ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"Validação:  {X_val.shape[0]} amostras  ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"Teste:      {X_test.shape[0]} amostras  ({X_test.shape[0]/len(X)*100:.0f}%)")
    print("Divisão concluída.")

    return X_train, X_val, X_test, y_train, y_val, y_test
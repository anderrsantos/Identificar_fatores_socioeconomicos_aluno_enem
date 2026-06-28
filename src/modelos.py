import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def plot_matriz_confusao(res: dict, pasta: str = "data/img") -> None:
    """
    Gera e salva a matriz de confusão normalizada [0-1] como imagem PNG.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(pasta, exist_ok=True)

    cm = res["matriz"].astype(float)
    cm_norm = cm / cm.sum()          # normaliza pelo total geral → [0, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    classes = ["< 600", "≥ 600"]
    ticks = [0, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {res['modelo']}")

    thresh = cm_norm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    nome_arquivo = res["modelo"].lower().replace(" ", "_").replace("á", "a").replace("ã", "a")
    caminho = os.path.join(pasta, f"matriz_{nome_arquivo}.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  Matriz de confusão salva em: {caminho}")


def print_metricas(res: dict):
    """Imprime as métricas de um dict retornado por calcular_metricas."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  MÉTRICAS — {res['modelo']}")
    print(f"{sep}")
    print(f"  Acurácia:   {res['acuracia']:.4f}")
    print(f"  Precisão:   {res['precisao']:.4f}")
    print(f"  Revocação:  {res['revocacao']:.4f}")
    print(f"  F1-Score:   {res['f1']:.4f}")
    print(f"  AUC-ROC:    {res['auc']:.4f}")
    print(f"  Matriz de Confusão:\n{res['matriz']}")
    plot_matriz_confusao(res)


def calcular_metricas(y_real, y_pred, y_prob, nome_modelo: str) -> dict:
    """
    Calcula todas as métricas exigidas pelo edital e as imprime via print_metricas.
    """
    acc  = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, zero_division=0)
    rec  = recall_score(y_real, y_pred, zero_division=0)
    f1   = f1_score(y_real, y_pred, zero_division=0)
    auc  = roc_auc_score(y_real, y_prob)
    cm   = confusion_matrix(y_real, y_pred)

    return {
        "modelo":    nome_modelo,
        "acuracia":  acc,
        "precisao":  prec,
        "revocacao": rec,
        "f1":        f1,
        "auc":       auc,
        "matriz":    cm,
    }


def tabela_comparativa(resultados: list[dict], etapa: str = "Validação") -> pd.DataFrame:
    cols = ["modelo", "acuracia", "precisao", "revocacao", "f1", "auc"]
    df = pd.DataFrame(resultados)[cols]
    df = df.set_index("modelo")
    df.columns = ["Acurácia", "Precisão", "Revocação", "F1-Score", "AUC-ROC"]

    print(f"\n{'='*65}")
    print(f"  TABELA COMPARATIVA — {etapa.upper()}")
    print(f"{'='*65}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 65)
    return df


class ModeloArvoreDecisao:
    """
    Árvore de Decisão (CART) via DecisionTreeClassifier.
      max_depth         – profundidade máxima da árvore
      min_samples_split – mínimo de amostras para dividir um nó
    """
    def __init__(self, max_depth: int = 8, min_samples_split: int = 10):
        self.nome   = "Árvore de Decisão"
        self.modelo = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

    def treinar(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)

    def avaliar(self, X, y) -> dict:
        y_pred = self.modelo.predict(X)
        y_prob = self.modelo.predict_proba(X)[:, 1]
        return calcular_metricas(y, y_pred, y_prob, self.nome)


class ModeloRandomForest:
    """
    Random Forest via RandomForestClassifier.
      n_estimators – número de árvores no ensemble
      max_depth    – profundidade máxima de cada árvore
    """
    def __init__(self, n_estimators: int = 150, max_depth: int = 10):
        self.nome   = "Random Forest"
        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    def treinar(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)

    def avaliar(self, X, y) -> dict:
        y_pred = self.modelo.predict(X)
        y_prob = self.modelo.predict_proba(X)[:, 1]
        return calcular_metricas(y, y_pred, y_prob, self.nome)


class ModeloRedeNeuralMLP:
    """
    Perceptron Multicamadas (MLP) via MLPClassifier.
      hidden_layer_sizes – tupla com nº de neurônios por camada oculta
      max_iter           – número máximo de épocas de treinamento
      learning_rate_init – taxa de aprendizado inicial (Adam)
    """
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=150, learning_rate_init=0.001):
        self.nome   = "Rede Neural MLP"
        self.modelo = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            activation="relu",
            solver="adam",
            random_state=42,
        )

    def treinar(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)

    def avaliar(self, X, y) -> dict:
        y_pred = self.modelo.predict(X)
        y_prob = self.modelo.predict_proba(X)[:, 1]
        return calcular_metricas(y, y_pred, y_prob, self.nome)
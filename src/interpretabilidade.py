# ============================================================
# interpretabilidade.py
# Análise de fatores socioeconômicos via Feature Importance
# (Árvore de Decisão + Random Forest) e SHAP (três modelos).
# ============================================================

import pandas as pd

DESCRICOES = {
    # Variáveis binárias diretas
    "TP_SEXO"            : "Sexo (M=1 / F=0)",
    "IN_TREINEIRO"       : "Treineiro (1 = só para treinar / 0 = candidato regular)",
    "Q025"               : "Acesso à internet em casa (0=Não / 1=Sim)",
    # TP_FAIXA_ETARIA — one-hot
    "TP_FAIXA_ETARIA_2"  : "Faixa etária: 15 a 17 anos",
    "TP_FAIXA_ETARIA_3"  : "Faixa etária: 18 anos",
    "TP_FAIXA_ETARIA_4"  : "Faixa etária: 19 anos",
    "TP_FAIXA_ETARIA_5"  : "Faixa etária: 20 a 24 anos",
    "TP_FAIXA_ETARIA_6"  : "Faixa etária: 25 a 29 anos",
    "TP_FAIXA_ETARIA_7"  : "Faixa etária: 30 a 34 anos",
    "TP_FAIXA_ETARIA_8"  : "Faixa etária: 35 a 39 anos",
    "TP_FAIXA_ETARIA_9"  : "Faixa etária: 40 a 44 anos",
    "TP_FAIXA_ETARIA_10" : "Faixa etária: 45 a 49 anos",
    "TP_FAIXA_ETARIA_11" : "Faixa etária: 50 a 54 anos",
    "TP_FAIXA_ETARIA_12" : "Faixa etária: 55 a 59 anos",
    "TP_FAIXA_ETARIA_13" : "Faixa etária: 60 a 64 anos",
    "TP_FAIXA_ETARIA_14" : "Faixa etária: 65 anos ou mais",
    # TP_COR_RACA — one-hot
    "TP_COR_RACA_2"      : "Cor/Raça: Preta",
    "TP_COR_RACA_3"      : "Cor/Raça: Parda",
    "TP_COR_RACA_4"      : "Cor/Raça: Amarela",
    "TP_COR_RACA_5"      : "Cor/Raça: Indígena",
    # TP_ESCOLA — tipo de escola — one-hot
    "TP_ESCOLA_2"        : "Escola: Pública Federal",
    "TP_ESCOLA_3"        : "Escola: Pública Estadual",
    "TP_ESCOLA_4"        : "Escola: Pública Municipal",
    "TP_ESCOLA_5"        : "Escola: Privada",
    # Q001 — escolaridade do pai — one-hot
    "Q001_B"             : "Pai: ensino fundamental incompleto",
    "Q001_C"             : "Pai: ensino fundamental completo",
    "Q001_D"             : "Pai: ensino médio completo",
    "Q001_E"             : "Pai: ensino superior completo",
    "Q001_F"             : "Pai: pós-graduação",
    # Q002 — escolaridade da mãe — one-hot
    "Q002_B"             : "Mãe: ensino fundamental incompleto",
    "Q002_C"             : "Mãe: ensino fundamental completo",
    "Q002_D"             : "Mãe: ensino médio completo",
    "Q002_E"             : "Mãe: ensino superior completo",
    "Q002_F"             : "Mãe: pós-graduação",
    # Q006 — renda familiar mensal — one-hot
    "Q006_B"             : "Renda familiar: até R$ 1.320",
    "Q006_C"             : "Renda familiar: R$ 1.320 – R$ 1.980",
    "Q006_D"             : "Renda familiar: R$ 1.980 – R$ 2.640",
    "Q006_E"             : "Renda familiar: R$ 2.640 – R$ 3.300",
    "Q006_F"             : "Renda familiar: R$ 3.300 – R$ 3.960",
    "Q006_G"             : "Renda familiar: R$ 3.960 – R$ 5.280",
    "Q006_H"             : "Renda familiar: R$ 5.280 – R$ 6.600",
    "Q006_I"             : "Renda familiar: R$ 6.600 – R$ 7.920",
    "Q006_J"             : "Renda familiar: R$ 7.920 – R$ 9.240",
    "Q006_K"             : "Renda familiar: R$ 9.240 – R$ 10.560",
    "Q006_L"             : "Renda familiar: acima de R$ 10.560",
}


class AnaliseInterpretabilidade:
    """
    Analisa quais fatores socioeconômicos mais impactam a
    predição de desempenho na redação do ENEM, usando dois métodos:

      1. Feature Importance nativa (DT e RF) — rápido, sem dependências extras.
      2. SHAP (DT, RF e MLP)               — mais robusto, requer `pip install shap`.
    """

    def __init__(self, modelo_dt, modelo_rf, modelo_mlp):
        self.modelo_dt  = modelo_dt
        self.modelo_rf  = modelo_rf
        self.modelo_mlp = modelo_mlp

    def _adicionar_descricao(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Descrição"] = df.index.map(lambda c: DESCRICOES.get(c, c))
        return df


    def shap(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        n_shap: int = 100,
        n_background: int = 50,
        top_n: int = 15,
    ) -> pd.DataFrame:
        try:
            import shap
        except ImportError:
            print("\n[AVISO] Biblioteca 'shap' não encontrada.")
            print("Instale com:  pip install shap")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("  Análise SHAP — DT + RF + MLP")
        print("=" * 60)

        X_shap       = X_test.sample(n=min(n_shap, len(X_test)), random_state=42)
        X_background = shap.sample(X_train, n_background, random_state=42)

        print("\n>>> SHAP — Árvore de Decisão...")
        exp_dt  = shap.TreeExplainer(self.modelo_dt.modelo)
        sv_dt   = exp_dt.shap_values(X_shap)
        vals_dt = sv_dt[1] if isinstance(sv_dt, list) else (sv_dt[:, :, 1] if sv_dt.ndim == 3 else sv_dt)
        df_dt   = pd.Series(abs(vals_dt).mean(axis=0), index=X_shap.columns, name="shap_dt")
        print("Top 10:")
        print(df_dt.sort_values(ascending=False).head(10).to_string(float_format=lambda x: f"{x:.4f}"))

        print("\n>>> SHAP — Random Forest...")
        exp_rf  = shap.TreeExplainer(self.modelo_rf.modelo)
        sv_rf   = exp_rf.shap_values(X_shap)
        vals_rf = sv_rf[1] if isinstance(sv_rf, list) else (sv_rf[:, :, 1] if hasattr(sv_rf, "ndim") and sv_rf.ndim == 3 else sv_rf)
        df_rf   = pd.Series(abs(vals_rf).mean(axis=0), index=X_shap.columns, name="shap_rf")
        print("Top 10:")
        print(df_rf.sort_values(ascending=False).head(10).to_string(float_format=lambda x: f"{x:.4f}"))

        print("\n>>> SHAP — Rede Neural MLP (pode demorar alguns minutos)...")
        exp_mlp  = shap.KernelExplainer(self.modelo_mlp.modelo.predict_proba, X_background)
        sv_mlp   = exp_mlp.shap_values(X_shap, nsamples=n_shap)
        vals_mlp = sv_mlp[1] if isinstance(sv_mlp, list) else (sv_mlp[:, :, 1] if hasattr(sv_mlp, "ndim") and sv_mlp.ndim == 3 else sv_mlp)
        df_mlp   = pd.Series(abs(vals_mlp).mean(axis=0), index=X_shap.columns, name="shap_mlp")
        print("Top 10:")
        print(df_mlp.sort_values(ascending=False).head(10).to_string(float_format=lambda x: f"{x:.4f}"))

        df_consolidado = pd.concat([df_dt, df_rf, df_mlp], axis=1)
        df_consolidado["shap_medio_geral"] = df_consolidado.mean(axis=1)
        df_consolidado = df_consolidado.sort_values("shap_medio_geral", ascending=False)
        df_consolidado = self._adicionar_descricao(df_consolidado)

        print(f"\n{'='*60}")
        print(f"  Ranking SHAP Consolidado (média DT + RF + MLP) — Top {top_n}")
        print(f"{'='*60}")
        print(df_consolidado.head(top_n).to_string(float_format=lambda x: f"{x:.4f}"))

        df_consolidado.to_csv("data/shap_fatores_socioeconomicos.csv")
        print("\nExportado: shap_fatores_socioeconomicos.csv")
        return df_consolidado
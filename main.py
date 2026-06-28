import pandas as pd

from src.leitura_arquivos   import leitura
from src.processamento      import pre_processamento
from src.divisao            import dividir
from src.modelos            import (
    ModeloArvoreDecisao,
    ModeloRandomForest,
    ModeloRedeNeuralMLP,
    tabela_comparativa,
    print_metricas,
)
from src.interpretabilidade import AnaliseInterpretabilidade


print("=" * 60)
print("Predição de desempenho na redação (MG)")
print("=" * 60)

# Leitura do dataset
dados = leitura()

# Pré-processamento
X, y, dados_processados = pre_processamento(dados)

print(f"\nBalanceamento do alvo:")
print(y.value_counts(normalize=True).rename({0: "< 600", 1: ">= 600"}).to_string())

# Divisão — 60% Treino | 20% Validação | 20% Teste
X_train, X_val, X_test, y_train, y_val, y_test = dividir(X, y)

# Instancia modelos
modelo_dt  = ModeloArvoreDecisao(max_depth=8,  min_samples_split=10)
modelo_rf  = ModeloRandomForest(n_estimators=150, max_depth=10)
modelo_mlp = ModeloRedeNeuralMLP(hidden_layer_sizes=(64, 32), max_iter=150, learning_rate_init=0.001)

# ============================================================
# Treinamento e Avaliação na VALIDAÇÃO
# ============================================================
print("\nTreinando...")

print("\n>>> Treinando Árvore de Decisão...")
modelo_dt.treinar(X_train, y_train)
res_dt_val = modelo_dt.avaliar(X_val, y_val)
#print_metricas(res_dt_val)

print("\n>>> Treinando Random Forest...")
modelo_rf.treinar(X_train, y_train)
res_rf_val = modelo_rf.avaliar(X_val, y_val)
#print_metricas(res_rf_val)

print("\n>>> Treinando Rede Neural MLP...")
modelo_mlp.treinar(X_train, y_train)
res_mlp_val = modelo_mlp.avaliar(X_val, y_val)
#print_metricas(res_mlp_val)

df_val = tabela_comparativa([res_dt_val, res_rf_val, res_mlp_val], etapa="Validação")

# ============================================================
# Avaliação final no TESTE
# ============================================================
print("\nTeste...")

print("\n>>> Avaliando no Teste — Árvore de Decisão...")
res_dt_test = modelo_dt.avaliar(X_test, y_test)
print_metricas(res_dt_test)

print("\n>>> Avaliando no Teste — Random Forest...")
res_rf_test = modelo_rf.avaliar(X_test, y_test)
print_metricas(res_rf_test)

print("\n>>> Avaliando no Teste — Rede Neural MLP...")
res_mlp_test = modelo_mlp.avaliar(X_test, y_test)
print_metricas(res_mlp_test)

df_test = tabela_comparativa([res_dt_test, res_rf_test, res_mlp_test], etapa="Teste")

# Exporta resultados
df_val.to_csv("data/resultados_validacao.csv")
df_test.to_csv("data/resultados_teste.csv")
print("\nResultados exportados: data/resultados_validacao.csv | data/resultados_teste.csv")

# ============================================================
# ANÁLISE DE INTERPRETABILIDADE
# ============================================================
analise = AnaliseInterpretabilidade(modelo_dt, modelo_rf, modelo_mlp)
analise.shap(X_train, X_test)
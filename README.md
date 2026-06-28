# ENEM 2023 — Predição de Desempenho na Redação (MG)
## Inteligência Computacional — CEFET-MG 01/2026

Projeto de classificação binária que prediz se um estudante mineiro
obteve nota **≥ 600** na redação do ENEM 2023, comparando três técnicas
de Inteligência Computacional: Árvore de Decisão, Random Forest e MLP.

---

## Estrutura do Projeto

```
.
├── main.py                   # Pipeline principal (ponto de entrada)
├── src/
│   ├── __init__.py
│   ├── dependencias.py       # Parâmetros e caminhos globais
│   ├── leitura_arquivos.py   # Download e leitura do dataset via Kaggle
│   ├── processamento.py      # Pré-processamento, encoding e normalização
│   ├── divisao.py            # Divisão treino/val/teste
│   ├── modelos.py            # Classes dos modelos, métricas e matriz de confusão
│   └── interpretabilidade.py # Análise SHAP
├── data/
│   ├── enem_2023_completo.csv        # Cache do dataset completo (gerado automaticamente)
│   ├── enem_2023_mg_amostra_40000.csv # Amostra MG utilizada nos experimentos
│   ├── resultados_validacao.csv      # Métricas na validação
│   ├── resultados_teste.csv          # Métricas no teste
│   ├── shap_fatores_socioeconomicos.csv
│   └── img/
│       ├── matriz_arvore_de_decisao.png
│       ├── matriz_random_forest.png
│       └── matriz_rede_neural_mlp.png
└── README.md
```

---

## Dataset

O projeto utiliza os microdados do ENEM 2023 disponíveis no Kaggle:

**[llssatcesar/microdados-enem-2023](https://www.kaggle.com/datasets/llssatcesar/microdados-enem-2023)**

Na **primeira execução**, o dataset é baixado automaticamente via `kagglehub`
e salvo em `data/enem_2023_completo.csv`. Nas execuções seguintes, o arquivo
local é utilizado diretamente, sem necessidade de nova conexão.


---

## Instalação das Dependências

É recomendado o uso de um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

Instale os pacotes:

```bash
pip install pandas scikit-learn kagglehub matplotlib shap
```

### Versões testadas

| Biblioteca   | Versão mínima |
|--------------|---------------|
| Python       | 3.10          |
| pandas       | 2.0           |
| scikit-learn | 1.3           |
| kagglehub    | 0.3           |
| matplotlib   | 3.7           |
| shap         | 0.44          |

---

## Como Executar

Execute a partir da **raiz do projeto** (pasta que contém `main.py`):

```bash
python main.py
```

O script irá:

1. Baixar o dataset do Kaggle (somente na primeira execução) e filtrar por MG
2. Pré-processar os dados (encoding, normalização MinMax, variável alvo binária)
3. Exibir o balanceamento das classes (< 600 / ≥ 600)
4. Dividir em **Treino 60% / Validação 20% / Teste 20%** com estratificação
5. Treinar e avaliar os três modelos na validação
6. Avaliar os modelos no conjunto de teste (avaliação final)
7. Exibir tabelas comparativas de validação e teste
8. Salvar matrizes de confusão normalizadas em `data/img/`
9. Executar análise SHAP e salvar ranking em `data/shap_fatores_socioeconomicos.csv`

---

## Variáveis Utilizadas

| Variável        | Descrição                              |
|-----------------|----------------------------------------|
| `TP_FAIXA_ETARIA` | Faixa etária do participante         |
| `TP_SEXO`       | Sexo (M/F)                             |
| `TP_COR_RACA`   | Cor/raça declarada                     |
| `TP_ESCOLA`     | Tipo de escola (pública/privada)       |
| `IN_TREINEIRO`  | Participou só para treinar (0/1)       |
| `Q001`          | Escolaridade do pai                    |
| `Q002`          | Escolaridade da mãe                    |
| `Q006`          | Renda familiar mensal                  |
| `Q025`          | Acesso à internet em casa (0/1)        |
| `NU_NOTA_REDACAO` | **Variável alvo** (≥ 600 → classe 1) |

---

## Parâmetros Configuráveis

Edite `src/dependencias.py` para ajustar o dataset e o tamanho da amostra:

```python
kaggle_dataset = "llssatcesar/microdados-enem-2023"
path_dataset   = "data/enem_2023_completo.csv"
n_amostra      = 40000
```

Edite `main.py` para alterar os hiperparâmetros dos modelos:

```python
modelo_dt  = ModeloArvoreDecisao(max_depth=8, min_samples_split=10)
modelo_rf  = ModeloRandomForest(n_estimators=150, max_depth=10)
modelo_mlp = ModeloRedeNeuralMLP(hidden_layer_sizes=(64, 32), max_iter=150, learning_rate_init=0.001)
```

---

## Métricas Reportadas

Para cada modelo, são calculadas e exibidas:

| Métrica           | Descrição                                                    |
|-------------------|--------------------------------------------------------------|
| **Acurácia**      | Proporção de acertos geral                                   |
| **Precisão**      | Dos classificados como ≥ 600, quantos realmente eram         |
| **Revocação**     | Dos que realmente eram ≥ 600, quantos foram identificados    |
| **F1-Score**      | Média harmônica de precisão e revocação                      |
| **AUC-ROC**       | Capacidade discriminativa geral do modelo                    |
| **Matriz de Confusão** | Normalizada [0-1], salva como imagem em `data/img/`    |

---

## Observações

- A normalização MinMax é ajustada **somente no treino** (`fit_transform`)
  e aplicada (`transform`) na validação e no teste.
- `random_state=42` é fixado em todos os pontos de aleatoriedade para
  garantir a **reprodutibilidade** dos experimentos.
- A análise SHAP pode demorar alguns minutos devido ao `KernelExplainer` da MLP.
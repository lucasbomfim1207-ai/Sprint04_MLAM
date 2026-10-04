# Challenge Sprint 4 — Classificação linear de `Energy_Class`

Dataset: [Renewable Energy Production Dataset (2010–2020)](https://www.kaggle.com/datasets/itsrohithere/renewable-energy-production-dataset-2010-2020/data)

## Integrantes


## Estrutura
| Caminho | Conteúdo |
|---|---|
| `dados/Renewable_Energy_Data.csv` | Base de dados utilizada |
| `Sprint4_Classificacao_Energia.ipynb` | Notebook completo (código + análises escritas + saídas) |
| `analise_sprint4.py` | Mesmo código em script (`python analise_sprint4.py`) |
| `figuras/` | Matrizes de correlação (Pearson, Spearman, completa), matrizes de confusão, ablação, comparativos |
| `resultados/` | CSVs de correlação, seleção de features, métricas por cenário, variabilidade; `saida_execucao.txt` |
| `requirements.txt` | Dependências |

## Resumo dos resultados
- **Encoding:** Low = 0, Medium = 1, High = 2 (alvo ordinal).
- **Correlação:** apenas `Efficiency_Ratio` se relaciona com o alvo (Spearman 0,74; Pearson 0,21 subestimado por outliers; após `log`, Pearson 0,73).
- **Features selecionadas (4):** `log(Efficiency_Ratio)`, `Temperature_C`, `Wind_Speed_m_s`, `Solar_Radiation_kWh_m2` (CV 5×3: 0,917 vs 0,911 com todas as numéricas e 0,907 com categóricas).
- **Modelo:** Regressão Logística multinomial (fronteiras = hiperplanos), com `StandardScaler` e divisão estratificada, `random_state=42`.

| Cenário | Treino/Teste | Acurácia | Precisão (macro) | Recall (macro) |
|---|---|---|---|---|
| 1 | 60% / 40% | 0,9250 | 0,9374 | 0,9087 |
| 2 | 85% / 15% | 0,9000 | 0,9177 | 0,8914 |

Em 200 partições aleatórias, a acurácia média é ≈ 0,915 nos dois tamanhos de teste; com 15% o desvio é maior (0,022 vs 0,013) — a diferença entre cenários é ruído amostral.

## Como executar
**Google Colab (sem upload/download):** no notebook, edite a variável `URL_CSV` (1ª célula de código) com o link *raw* do CSV neste repositório:
`https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/dados/Renewable_Energy_Data.csv`
O `pd.read_csv` lê o arquivo direto da internet; se o link falhar, usa `dados/Renewable_Energy_Data.csv` local.

**Local:**
```bash
pip install -r requirements.txt
python analise_sprint4.py          # ou abra o notebook no Jupyter
```

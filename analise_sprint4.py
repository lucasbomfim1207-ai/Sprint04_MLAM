# -*- coding: utf-8 -*-
"""
Challenge Sprint 4 - Classificacao linear de Energy_Class
Dataset: Renewable Energy Production Dataset (2010-2020) - Kaggle

Integrantes (nome completo - RM):
  - Eduardo Barcelos De Carvalho Braziliano - 573274
  - Julia Johanson Peniche Dias Da Silva    - 572220
  - Lucas Bomfim Leite                      - 570420

Como executar (a partir da pasta raiz do repositorio):
    pip install -r requirements.txt
    python analise_sprint4.py

O CSV e lido direto do GitHub (URL_CSV); se o link falhar, usa dados/Renewable_Energy_Data.csv.
Graficos sao salvos em figuras/ e as tabelas em resultados/.
Para tambem ABRIR os graficos em janelas (VS Code/PyCharm), mude MOSTRAR_GRAFICOS para True.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
MOSTRAR_GRAFICOS = False   # True = abre as janelas dos graficos (bloqueia ate fechar cada uma)

import matplotlib
if not MOSTRAR_GRAFICOS:
    matplotlib.use("Agg")   # execucao sem janelas (so salva os arquivos)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, classification_report)

import os
SEED = 42
# Link RAW do CSV no GitHub (pd.read_csv le direto da internet, sem baixar nem upload)
URL_CSV = "https://raw.githubusercontent.com/lucasbomfim1207-ai/Sprint04_MLAM/refs/heads/main/dados/Renewable_Energy_Data.csv"
CAMINHO = "dados/Renewable_Energy_Data.csv"  # alternativa local


def fechar():
    """Finaliza a figura atual: mostra (se MOSTRAR_GRAFICOS) e fecha."""
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close()


for pasta in ("figuras", "resultados"):
    os.makedirs(pasta, exist_ok=True)

# ============================================================ 01) ANALISE DO DATASET
# a) Carregar
try:
    df = pd.read_csv(URL_CSV)
    print("Dataset carregado direto do link (GitHub raw).")
except Exception as e:
    print("Link indisponivel, usando arquivo local:", e)
    df = pd.read_csv(CAMINHO)
print("Shape:", df.shape)
print(df.dtypes)
print("Nulos:", df.isna().sum().sum(), "| Duplicados:", df.duplicated().sum())
print(df["Energy_Class"].value_counts())

# b) Encoding da variavel-alvo (ordinal: Low < Medium < High)
mapa = {"Low": 0, "Medium": 1, "High": 2}
df["Energy_Class_enc"] = df["Energy_Class"].map(mapa)

num_cols = ["Temperature_C", "Wind_Speed_m_s", "Solar_Radiation_kWh_m2", "Rainfall_mm",
            "Efficiency_Ratio", "Lagged_Production_MWh", "Combined_Weather_Index"]
cat_cols = ["Region", "Energy_Source", "Season"]


# c) Matrizes de correlacao
def heat(corr, titulo, arq, size=(9, 7)):
    plt.figure(figsize=size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=.5, cbar_kws={"label": "correlacao"})
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(arq, dpi=150)
    fechar()


num_y = df[num_cols + ["Energy_Class_enc"]]
corr_p = num_y.corr(method="pearson")
corr_s = num_y.corr(method="spearman")
heat(corr_p, "Matriz de correlacao de Pearson (numericas + Energy_Class codificada)", "figuras/01_correlacao_pearson.png")
heat(corr_s, "Matriz de correlacao de Spearman (numericas + Energy_Class codificada)", "figuras/02_correlacao_spearman.png")
corr_p.round(4).to_csv("resultados/correlacao_pearson.csv")
corr_s.round(4).to_csv("resultados/correlacao_spearman.csv")

# Matriz completa com categoricas (one-hot) para avaliar todas as features
df_oh = pd.get_dummies(df.drop(columns="Energy_Class"), columns=cat_cols).astype(float)
corr_full = df_oh.corr(method="spearman")
heat(corr_full, "Spearman - todas as features (categoricas em one-hot) + Energy_Class",
     "figuras/03_correlacao_spearman_completa.png", (15, 12))
corr_full.round(4).to_csv("resultados/correlacao_spearman_completa.csv")

tab_corr = pd.DataFrame({"Pearson": corr_p["Energy_Class_enc"], "Spearman": corr_s["Energy_Class_enc"]}).drop("Energy_Class_enc")
tab_corr["|Spearman|"] = tab_corr["Spearman"].abs()
tab_corr = tab_corr.sort_values("|Spearman|", ascending=False)
tab_corr.round(4).to_csv("resultados/correlacao_com_alvo.csv")
print("\nCorrelacao com o alvo:\n", tab_corr.round(3))
print("\nAssimetria (skew):\n", df[num_cols].skew().round(2))
print("\nCorrelacao categoricas (one-hot) x alvo (Spearman):\n",
      corr_full["Energy_Class_enc"].drop("Energy_Class_enc").sort_values(key=abs, ascending=False).round(3).head(8))

# Evidencia visual: outliers distorcem Pearson do Efficiency_Ratio
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.boxplot(data=df, x="Energy_Class", y="Efficiency_Ratio", order=["Low", "Medium", "High"], ax=ax[0])
ax[0].set_title("Efficiency_Ratio por classe (escala linear)")
sns.boxplot(data=df, x="Energy_Class", y="Efficiency_Ratio", order=["Low", "Medium", "High"], ax=ax[1])
ax[1].set_yscale("log")
ax[1].set_title("Efficiency_Ratio por classe (escala log)")
sns.histplot(df["Efficiency_Ratio"], bins=60, ax=ax[2])
ax[2].set_title("Distribuicao de Efficiency_Ratio (cauda longa)")
plt.tight_layout()
plt.savefig("figuras/04_efficiency_outliers.png", dpi=150)
fechar()

# Transformacoes (reduzem o efeito dos outliers; sao funcoes linha a linha, logo sem vazamento)
df["log_Efficiency_Ratio"] = np.log(df["Efficiency_Ratio"])
df["slog_Combined_Weather_Index"] = np.sign(df["Combined_Weather_Index"]) * np.log1p(df["Combined_Weather_Index"].abs())
print("Pearson alvo x log_Efficiency_Ratio:", round(df["log_Efficiency_Ratio"].corr(df["Energy_Class_enc"]), 3))

# d) Selecao de features: validacao cruzada repetida (5 folds x 3) com regressao logistica
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
y_all = df["Energy_Class_enc"]
todas_num = ["log_Efficiency_Ratio", "Temperature_C", "Wind_Speed_m_s", "Solar_Radiation_kWh_m2",
             "Rainfall_mm", "Lagged_Production_MWh", "slog_Combined_Weather_Index"]


def cv_score(nums, cats=()):
    trans = [("n", StandardScaler(), list(nums))]
    if cats:
        trans.append(("c", OneHotEncoder(), list(cats)))
    pipe = Pipeline([("p", ColumnTransformer(trans)), ("m", LogisticRegression(max_iter=5000))])
    s = cross_val_score(pipe, df[list(nums) + list(cats)], y_all, cv=cv)
    return s.mean(), s.std()


FEATURES = ["log_Efficiency_Ratio", "Temperature_C", "Wind_Speed_m_s", "Solar_Radiation_kWh_m2"]
brutas = ["Efficiency_Ratio", "Temperature_C", "Wind_Speed_m_s", "Solar_Radiation_kWh_m2",
          "Rainfall_mm", "Lagged_Production_MWh", "Combined_Weather_Index"]
opcoes = [
    ("Baseline (classe majoritaria)", None, ()),
    ("Somente Efficiency_Ratio bruto (maior |r| de Pearson)", ["Efficiency_Ratio"], ()),
    ("Somente log(Efficiency_Ratio)", ["log_Efficiency_Ratio"], ()),
    ("Todas as numericas brutas (sem transformacao)", brutas, ()),
    ("Todas as numericas (com log)", todas_num, ()),
    ("Todas as numericas (com log) + categoricas", todas_num, cat_cols),
    ("Subconjunto selecionado (4 features)", FEATURES, ()),
]
linhas = []
for nome, cols, cats in opcoes:
    if cols is None:
        linhas.append((nome, 0, y_all.value_counts(normalize=True).max(), 0.0))
        continue
    m, s = cv_score(cols, cats)
    linhas.append((nome, len(cols) + len(cats), m, s))
tab_sel = pd.DataFrame(linhas, columns=["Conjunto de features", "N features", "Acuracia CV (media)", "Desvio"])
tab_sel.round(4).to_csv("resultados/comparacao_selecao_features.csv", index=False)
print("\nSelecao de features (CV 5x3):\n", tab_sel.round(4).to_string(index=False))

# Ablacao: efeito de remover cada feature do conjunto completo
ref = cv_score(todas_num)[0]
abl = [(f, ref - cv_score([c for c in todas_num if c != f])[0]) for f in todas_num]
tab_abl = pd.DataFrame(abl, columns=["Feature removida", "Queda de acuracia"]).sort_values("Queda de acuracia", ascending=False)
tab_abl.round(4).to_csv("resultados/ablacao_features.csv", index=False)
print("\nAblacao (referencia = todas numericas):\n", tab_abl.round(4).to_string(index=False))
plt.figure(figsize=(8, 4))
sns.barplot(data=tab_abl, y="Feature removida", x="Queda de acuracia", color="steelblue")
plt.axvline(0, color="k", lw=.8)
plt.title("Queda de acuracia ao remover cada feature (CV 5x3)")
plt.tight_layout()
plt.savefig("figuras/05_ablacao_features.png", dpi=150)
fechar()

# ============================================================ 02) MODELO
# Regressao Logistica multinomial: score_k(x) = w_k . x + b_k  -> fronteiras = hiperplanos


def treinar_avaliar(test_size, nome):
    X, y = df[FEATURES], df["Energy_Class_enc"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000, random_state=SEED))])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
    res = {
        "Cenario": nome, "Treino": len(X_tr), "Teste": len(X_te),
        "Acuracia": accuracy_score(y_te, pred),
        "Precisao (macro)": precision_score(y_te, pred, average="macro"),
        "Recall (macro)": recall_score(y_te, pred, average="macro"),
        "Precisao (ponderada)": precision_score(y_te, pred, average="weighted"),
        "Recall (ponderado)": recall_score(y_te, pred, average="weighted"),
    }
    nomes = ["Low", "Medium", "High"]
    rel = classification_report(y_te, pred, target_names=nomes, digits=4, output_dict=True)
    print(f"\n===== {nome} =====\n", classification_report(y_te, pred, target_names=nomes, digits=4))
    print("Matriz de confusao (linhas = real, colunas = previsto):\n", cm)
    return pipe, cm, res, rel


pipe1, cm1, res1, rel1 = treinar_avaliar(0.40, "Cenario 1 (60/40)")
pipe2, cm2, res2, rel2 = treinar_avaliar(0.15, "Cenario 2 (85/15)")

coef = pd.DataFrame(pipe2.named_steps["m"].coef_, columns=FEATURES, index=["Low", "Medium", "High"])
coef["intercepto"] = pipe2.named_steps["m"].intercept_
coef.round(4).to_csv("resultados/coeficientes_modelo_cenario2.csv")
print("\nCoeficientes (cenario 2, features padronizadas):\n", coef.round(3))

# ============================================================ 03) AVALIACAO
for cm, tit, arq in [(cm1, "Matriz de confusao - Cenario 1 (60% treino / 40% teste)", "figuras/06_confusao_cenario1.png"),
                     (cm2, "Matriz de confusao - Cenario 2 (85% treino / 15% teste)", "figuras/07_confusao_cenario2.png")]:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(tit, fontsize=10)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    plt.tight_layout()
    plt.savefig(arq, dpi=150)
    fechar()

met = pd.DataFrame([res1, res2]).set_index("Cenario")
met.round(4).to_csv("resultados/metricas_comparativo.csv")
print("\nMetricas:\n", met.round(4).T)

por_classe = []
for nome, rel in [("Cenario 1 (60/40)", rel1), ("Cenario 2 (85/15)", rel2)]:
    for c in ["Low", "Medium", "High"]:
        por_classe.append({"Cenario": nome, "Classe": c, "Precisao": rel[c]["precision"], "Recall": rel[c]["recall"],
                           "F1": rel[c]["f1-score"], "Suporte": rel[c]["support"]})
pc = pd.DataFrame(por_classe)
pc.round(4).to_csv("resultados/metricas_por_classe.csv", index=False)
print("\nPor classe:\n", pc.round(4).to_string(index=False))

plot = met[["Acuracia", "Precisao (macro)", "Recall (macro)"]].reset_index().melt(
    id_vars="Cenario", var_name="Metrica", value_name="Valor")
plt.figure(figsize=(8, 4.5))
ax = sns.barplot(data=plot, x="Metrica", y="Valor", hue="Cenario")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=8)
plt.ylim(0, 1.05)
plt.title("Comparativo de metricas por cenario")
plt.tight_layout()
plt.savefig("figuras/08_comparativo_metricas.png", dpi=150)
fechar()


# Variabilidade da estimativa: 200 particoes aleatorias para cada tamanho de teste
def distrib(test_size, n=200):
    accs = []
    for i in range(n):
        Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["Energy_Class_enc"], test_size=test_size,
                                              random_state=1000 + i, stratify=df["Energy_Class_enc"])
        p = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]).fit(Xtr, ytr)
        accs.append(accuracy_score(yte, p.predict(Xte)))
    return np.array(accs)


a40, a15 = distrib(0.40), distrib(0.15)
var = pd.DataFrame({"Teste": ["40%", "15%"], "Media": [a40.mean(), a15.mean()], "Desvio": [a40.std(), a15.std()],
                    "Min": [a40.min(), a15.min()], "Max": [a40.max(), a15.max()]})
var.round(4).to_csv("resultados/variabilidade_200_particoes.csv", index=False)
print("\nVariabilidade (200 particoes):\n", var.round(4).to_string(index=False))
plt.figure(figsize=(8, 4.5))
sns.kdeplot(a40, fill=True, label="Teste 40%")
sns.kdeplot(a15, fill=True, label="Teste 15%")
plt.xlabel("Acuracia")
plt.title("Distribuicao da acuracia em 200 particoes aleatorias")
plt.legend()
plt.tight_layout()
plt.savefig("figuras/09_variabilidade_acuracia.png", dpi=150)
fechar()
print("\nConcluido.")

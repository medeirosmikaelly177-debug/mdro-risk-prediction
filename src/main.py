
# Projeto: Predição de risco de MDRO em pacientes de UTI
# Protótipo metodológico baseado em estudo científico de 2024.
#
# ATENÇÃO: os dados utilizados são sintéticos.
# Os resultados não representam pacientes reais nem desempenho clínico.

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)


# 1. GERAÇÃO DA BASE SINTÉTICA

variaveis = [
    "idade",
    "pcr",
    "ureia",
    "dias_internacao",
    "frequencia_cardiaca",
    "temperatura",
    "pressao_arterial",
    "comorbidades"
]

X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    weights=[0.8804, 0.1196],
    random_state=42
)

df = pd.DataFrame(X, columns=variaveis)
df["mdro"] = y

print("QUANTIDADE DE REGISTROS")
print(len(df))

print("\nDISTRIBUICAO DAS CLASSES")
print(df["mdro"].value_counts().sort_index())


# 2. ANALISE DE CORRELACAO - SCIPY

print("\nCORRELACAO DE SPEARMAN")

for coluna in variaveis:
    correlacao, p_valor = spearmanr(
        df[coluna],
        df["mdro"]
    )

    print(
        f"{coluna}: correlacao = {correlacao:.3f} | "
        f"p-valor = {p_valor:.4f}"
    )


# 3. SEPARACAO DOS DADOS

X = df[variaveis]
y = df["mdro"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# 4. NORMALIZACAO MIN-MAX

scaler = MinMaxScaler()

X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm = scaler.transform(X_teste)


# 5. DEFINICAO DOS MODELOS

modelos = {
    "Regressao Logistica": (
        LogisticRegression(max_iter=1000),
        X_treino_norm,
        X_teste_norm
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        X_treino,
        X_teste
    )
}

resultados = {}
modelos_treinados = {}


# 6. TREINAMENTO E AVALIACAO

for nome, (modelo, treino, teste) in modelos.items():

    modelo.fit(treino, y_treino)

    probabilidades = modelo.predict_proba(teste)[:, 1]
    previsoes = modelo.predict(teste)

    auc = roc_auc_score(y_teste, probabilidades)

    resultados[nome] = auc
    modelos_treinados[nome] = (modelo, teste)

    print(f"\nMODELO: {nome}")
    print(f"AUROC: {auc:.4f}")

    print("\nMatriz de confusao:")
    print(confusion_matrix(y_teste, previsoes))

    print("\nRelatorio de classificacao:")
    print(
        classification_report(
            y_teste,
            previsoes,
            target_names=["Sem MDRO", "MDRO"],
            zero_division=0
        )
    )


# 7. COMPARACAO DOS MODELOS

print("\nCOMPARACAO FINAL")

for nome, auc in resultados.items():
    print(f"{nome}: {auc:.4f}")


# 8. GRAFICO DA CURVA ROC

plt.figure(figsize=(8, 5))

for nome, (modelo, teste) in modelos_treinados.items():

    RocCurveDisplay.from_estimator(
        modelo,
        teste,
        y_teste,
        name=nome,
        ax=plt.gca()
    )

plt.title("Comparacao dos modelos - MDRO")
plt.tight_layout()
plt.show()


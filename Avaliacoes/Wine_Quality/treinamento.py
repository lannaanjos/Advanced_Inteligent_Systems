'''
QUESTÃO 2 (1,0 ponto)
Wine Quality
https://archive.ics.uci.edu/dataset/186/wine+quality

Tarefa: Treinar um classificador em um módulo de inferência para demonstrar o modelo treinado em funcionamento.
Monte um arquivo com todos os dados (vinhos tintos e brancos) com os nomes das colunas.
O atributo alvo (classe) é a coluna quality.

Avalie ao menos três metaestimadores classificadores e selecione aquele com o melhor desempenho.
Você deve demonstrar:
        a)      O fluxo de procedimentos até o treinamento do modelo (pipeline)
        b)      Acurácia global, acurácia por classes (use a matriz de confusão para tanto) e a medida f1-score
        c)      Justificar qual é o modelo mais adequado para possível implantação

Modelo mais adequado: Random Forest

/\/ JUSTIFICATIVA:
O RF obteve a maior acurácia global (0.64) o e melhor F1 Score ponderado (0.64) entre os três modelos
avaliados.Ele é mais adequado pois é um ensemble de árvores que reduz overfitting por agregação, enquanto
a Decision Tree tenta memorizar.
Ele também lida melhor com o resíduo desbalanceado de SMOTE, já que ele tem um recall mais balanceado entre
as lasses majoritárias e minoritárias.
Além disso, sua robustez a ruído e outliers é relevante num dataset de avaliação sensorial subjetiva
como o esse.
'''

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_PATH = Path(__file__).parent

DATA_RED_PATH = BASE_PATH.parent.parent / "Data" / "winequality-red.csv"
DATA_WHITE_PATH = BASE_PATH.parent.parent / "Data" / "winequality-white.csv"

NORM_PATH = BASE_PATH / "norm_wq.pkl"
COLUNAS_PATH = BASE_PATH / "colunas_wq.pkl"

RF_PATH = BASE_PATH / "rf_wq.pkl"
DT_PATH = BASE_PATH / "dt_wq.pkl"
KNN_PATH = BASE_PATH / "knn_wq.pkl"

CM_RF_PATH = BASE_PATH / "cm_rf_wq.png"
CM_DT_PATH = BASE_PATH / "cm_dt_wq.png"
CM_KNN_PATH = BASE_PATH / "cm_knn_wq.png"

RF_GRID = {
    "n_estimators":      [int(x) for x in np.linspace(10, 100, 10)],
    "criterion":         ["gini", "entropy"],
    "min_samples_split": [int(x) for x in np.linspace(2, 10, 2)],
    "max_depth":         [int(x) for x in np.linspace(10, 100, 20)],
    "max_features":      ["sqrt", "log2"]
}

DT_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [int(x) for x in np.linspace(5, 50, 10)],
    "min_samples_split": [int(x) for x in np.linspace(2, 20, 5)],
    "min_samples_leaf": [1, 2, 4, 8]
}

KNN_GRID = {
    "n_neighbors": [int(x) for x in np.linspace(3, 30, 10)],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}


# /\ UTILS
# /\/\ CARREGAMENTO E COMBINAÇÃO
def carregar_dados() -> pd.DataFrame:
    df_red = pd.read_csv(DATA_RED_PATH,   sep=";")
    df_white = pd.read_csv(DATA_WHITE_PATH, sep=";")

    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"

    df = pd.concat([df_red, df_white], ignore_index=True)

    print(f"Dataset combinado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"Distribuição de quality:\n{df['quality'].value_counts().sort_index()}\n")

    return df


# /\/\ PRE-PROCESSAMENTO
def preprocessamento(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, MinMaxScaler, list]:
    classe = df["quality"]
    atributos = df.drop(columns=["quality"])

    # wine_type e categorica, todas as demais sao numericas continuas
    dados_cat = pd.get_dummies(atributos[["wine_type"]], prefix_sep="_", dtype=int)
    dados_num = atributos.drop(columns=["wine_type"])

    scaler = MinMaxScaler()
    dados_num_norm = scaler.fit_transform(dados_num)
    dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

    atributos_prontos = dados_num_norm.join(dados_cat.reset_index(drop=True))
    colunas = atributos_prontos.columns.tolist()

    pickle.dump(scaler, open(NORM_PATH, "wb"))
    pickle.dump(colunas, open(COLUNAS_PATH, "wb"))

    print(f"Shape após pré-processamento: {atributos_prontos.shape}")
    print(f"Scaler e colunas salvos!\n")

    return atributos_prontos, classe, scaler, colunas


# /\/\ METRICAS
def calcular_metricas(nome: str, y_true, y_pred, classes, modelo, x_teste, cm_path: Path):
    print(f"MÉTRICAS -- {nome}")

    acuracia = accuracy_score(y_true, y_pred)
    print(f"Acurácia global: {acuracia:.4f}")

    print("\nRelatório por classe (F1, sensibilidade):")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0))

    # especificidade por classe: TN / (TN + FP) -- calculo manual p multiclasse
    cm = confusion_matrix(y_true, y_pred)
    especificidades = []
    for i in range(len(classes)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        especificidades.append(esp)
        print(f"  Especificidade [{classes[i]}]: {esp:.4f}")

    print(f"  Especificidade média (macro): {np.mean(especificidades):.4f}\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        modelo, x_teste, y_true,
        display_labels=[str(c) for c in classes],
        ax=ax, xticks_rotation="vertical"
    )
    plt.title(f"Confusion Matrix -- {nome}")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"  Matriz salva em {cm_path}\n")


# /\/\ TREINAMENTO
def treinar_modelo(nome: str, estimador, grid: dict, atributos, classe, pkl_path: Path, cm_path: Path):
    x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors = 1)
    x_treino, y_treino = smote.fit_resample(x_treino, y_treino)

    print(f"[{nome}] Shape treino após SMOTE: {x_treino.shape}")

    search = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=grid,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(x_treino, y_treino)

    print(f"\n[{nome}] Melhores parâmetros: {search.best_params_}")

    melhor_modelo = search.best_estimator_
    y_pred = melhor_modelo.predict(x_teste)
    classes = melhor_modelo.classes_

    calcular_metricas(nome, y_teste, y_pred, classes, melhor_modelo, x_teste, cm_path)

    pickle.dump(melhor_modelo, open(pkl_path, "wb"))
    print(f"[{nome}] Modelo salvo em {pkl_path}\n")

    return melhor_modelo, accuracy_score(y_teste, y_pred)


def main():
    df = carregar_dados()

    atributos, classe, _, _ = preprocessamento(df)

    resultados = {}

    modelo_rf, acc_rf = treinar_modelo(
        "Random Forest",
        RandomForestClassifier(random_state=42), RF_GRID,
        atributos, classe, RF_PATH, CM_RF_PATH
    )
    resultados["Random Forest"] = acc_rf

    modelo_dt, acc_dt = treinar_modelo(
        "Decision Tree",
        DecisionTreeClassifier(random_state=42), DT_GRID,
        atributos, classe, DT_PATH, CM_DT_PATH
    )
    resultados["Decision Tree"] = acc_dt

    modelo_knn, acc_knn = treinar_modelo(
        "KNN",
        KNeighborsClassifier(), KNN_GRID,
        atributos, classe, KNN_PATH, CM_KNN_PATH
    )
    resultados["KNN"] = acc_knn

    print("COMPARAÇÃO DE MODELOS")
    for nome, acc in sorted(resultados.items(), key=lambda x: -x[1]):
        print(f"  {nome:<20} Acurácia: {acc:.4f}")

    melhor = max(resultados, key=resultados.get)
    print(f"\n  Melhor modelo: {melhor} ({resultados[melhor]:.4f})")
    print("="*50)


if __name__ == "__main__":
    main()

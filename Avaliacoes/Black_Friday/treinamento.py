'''
QUESTÃO 3 (1,0 ponto)
Black Friday Sales Dataset
https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset

Tarefa: Treinar um classificador de acordo com as discussões em sala de aula em um módulo de inferência para demonstrar o modelo treinado em funcionamento
O seu sistema inteligente deve:
indicar qual é a categoria de produto (product category) de acordo com a circunstância na qual uma venda ocorra;
indicar qual será a forma de pagamento (payment_method) mais provável de acordo com a circunstância da venda
indicar qual é o faixa etária (age_group) mais provável do comprador de acordo com a circunstância da venda;

Para todas as indicações acima, o seu sistema de demonstrar um grau de certeza, com base no score da classificação
Você deve demonstrar, para cada indicação do seu sistema inteligente:
        a)      O fluxo de procedimentos até o treinamento do modelo (pipeline);
        b)      Acurácia global, acurácia por classes (use a matriz de confusão para tanto) e a medida f1-score;
        c)      O funcionamento do sistema inteligente inficando categoria do produto, forma de pagamento e faixa etária


Medidas de acurácia a serem computadas e demonstradas
Acurácia global
Especificidade
Sensibilidade
F1-Score
'''


from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_PATH = Path(__file__).parent
 
DATA_PATH = BASE_PATH.parent.parent / "Data" / "retail_black_friday_sales_100k.csv"
NORM_PATH = BASE_PATH / "norm_bf.pkl"
COLUNAS_PATH = BASE_PATH / "colunas_bf.pkl"
 
RF_PRODUCT_CATEGORY_PATH = BASE_PATH / "rf_product_category.pkl"
RF_PAYMENT_METHOD_PATH = BASE_PATH / "rf_payment_method.pkl"
RF_AGE_GROUP_PATH = BASE_PATH / "rf_age_group.pkl"
 
CM_PRODUCT_CATEGORY_PATH = BASE_PATH / "cm_product_category.png"
CM_PAYMENT_METHOD_PATH = BASE_PATH / "cm_payment_method.png"
CM_AGE_GROUP_PATH = BASE_PATH / "cm_age_group.png"


BLACK_FRIDAY = pd.Timestamp("2025-11-29")

# descartando os ids
# mantendo purchase date porque pessoas que compram antecipandamente e no pico da black friday podem possuir diferentes perfis
COLUNAS_DROP = ["transaction_id", "customer_id", "product_id"]
COLUNAS_CAT = ["gender", "city", "customer_segment"]
COLUNAS_NUM = ["original_price", "discount_pct", "final_price", "quantity", "purchase_amount", "purchase_hour", "days_to_black_friday"]
COLUNAS_BIN = ["is_weekend", "is_black_friday"]
TARGETS = ["product_category", "payment_method", "age_group"]

RF_GRID = {
    "n_estimators": [int(x) for x in np.linspace(10, 100, 10)],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [int(x) for x in np.linspace(2, 10, 2)],
    "max_depth": [int(x) for x in np.linspace(10, 100, 20)],
    "max_features": ["sqrt", "log2"]
}

# /\ UTILS
def preprocessamento(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, list]:
    df = df.drop(columns=COLUNAS_DROP)

    # distancia absoluta em dias ate a Black Friday
    df["days_to_black_friday"] = (pd.to_datetime(df["purchase_date"]) - BLACK_FRIDAY).dt.days.abs()
    df = df.drop(columns=["purchase_date"])

    # separa targets
    targets = df[TARGETS]
    df = df.drop(columns=TARGETS)

    dados_num = df[COLUNAS_NUM]
    scaler = MinMaxScaler()
    dados_num_norm = scaler.fit_transform(dados_num)
    dados_num_norm = pd.DataFrame(dados_num_norm, columns=COLUNAS_NUM)

    dados_cat = pd.get_dummies(df[COLUNAS_CAT], prefix_sep="_", dtype=int)
    dados_bin = df[COLUNAS_BIN].reset_index(drop=True)

    atributos = dados_num_norm.join(dados_cat.reset_index(drop=True)).join(dados_bin)
    colunas = atributos.columns.tolist()

    pickle.dump(scaler, open(NORM_PATH, "wb"))
    pickle.dump(colunas, open(COLUNAS_PATH, "wb"))

    print(f"Shape atributos: {atributos.shape}")
    print(f"Scaler e colunas salvos!")

    return atributos, targets, scaler, colunas

# /\/\
def calcular_metricas(nome: str, y_true, y_pred, classes, modelo, x_teste, cm_path: Path):
    print(f"MÉTRICAS -- {nome}")

    acuracia = accuracy_score(y_true, y_pred)
    print(f"Acurácia global: {acuracia:.4f}")

    print("\nRelatório por classe (F1, sensibilidade):")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))

    # especificidade por classe: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    especificidades = []
    for i in range(len(classes)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        especificidades.append(esp)
        print(f" Especificidade [{classes[i]}]: {esp:.4f}")

    print(f" Especificidade média (macro): {np.mean(especificidades):.4f}")

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
    print(f"  Matriz salva em {cm_path}")


def treinar_modelo(nome: str, atributos, target, pkl_path: Path, cm_path: Path) -> RandomForestClassifier:
    x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, target, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=RF_GRID,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    rf_search.fit(x_treino, y_treino)

    print(f"\n[{nome}] Melhores parâmetros: {rf_search.best_params_}")

    melhor_modelo = rf_search.best_estimator_
    y_pred = melhor_modelo.predict(x_teste)
    classes = melhor_modelo.classes_

    calcular_metricas(nome, y_teste, y_pred, classes, melhor_modelo, x_teste, cm_path)

    pickle.dump(melhor_modelo, open(pkl_path, "wb"))
    print(f"[{nome}] Modelo salvo em {pkl_path}")

    return melhor_modelo


def main():
    df = pd.read_csv(DATA_PATH, sep=",")
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    atributos, targets, _, _ = preprocessamento(df)

    treinar_modelo(
        "Product Category",
        atributos, targets["product_category"],
        RF_PRODUCT_CATEGORY_PATH, CM_PRODUCT_CATEGORY_PATH
    )

    treinar_modelo(
        "Payment Method",
        atributos, targets["payment_method"],
        RF_PAYMENT_METHOD_PATH, CM_PAYMENT_METHOD_PATH
    )

    treinar_modelo(
        "Age Group",
        atributos, targets["age_group"],
        RF_AGE_GROUP_PATH, CM_AGE_GROUP_PATH
    )

    print("\nTreinamento concluído para os 3 modelos!")


if __name__ == "__main__":
    main()

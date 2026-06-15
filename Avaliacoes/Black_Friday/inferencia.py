# Carrega os 3 modelos treinados e prediz, com grau de certeza:
#   - product_category: categoria do produto
#   - payment_method: forma de pagamento mais provavel
#   - age_group: faixa etaria do comprador

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent

NORM_PATH = BASE_PATH / "norm_bf.pkl"
COLUNAS_PATH = BASE_PATH / "colunas_bf.pkl"
RF_PRODUCT_CATEGORY_PATH = BASE_PATH / "rf_product_category.pkl"
RF_PAYMENT_METHOD_PATH = BASE_PATH / "rf_payment_method.pkl"
RF_AGE_GROUP_PATH = BASE_PATH / "rf_age_group.pkl"

BLACK_FRIDAY = pd.Timestamp("2025-11-29")

COLUNAS_CAT = ["gender", "city", "customer_segment"]
COLUNAS_NUM = ["original_price", "discount_pct", "final_price", "quantity",
               "purchase_amount", "purchase_hour", "days_to_black_friday"]
COLUNAS_BIN = ["is_weekend", "is_black_friday"]

# /\/\ CARREGANDO MODELOS
scaler = pickle.load(open(NORM_PATH, "rb"))
colunas = pickle.load(open(COLUNAS_PATH, "rb"))
modelo_category = pickle.load(open(RF_PRODUCT_CATEGORY_PATH, "rb"))
modelo_payment = pickle.load(open(RF_PAYMENT_METHOD_PATH, "rb"))
modelo_age = pickle.load(open(RF_AGE_GROUP_PATH, "rb"))

# /\/\ DADOS DA VENDA
venda = {
    "gender": "Female",
    "city": "Miami",
    "customer_segment": "VIP",
    "original_price": 399.61,
    "discount_pct": 40,
    "final_price": 239.77,
    "quantity": 1,
    "purchase_amount": 239.77,
    "purchase_hour": 23,
    "purchase_date": "2025-11-28",
    "is_weekend": 1,
    "is_black_friday": 0,
}

# /\/\ PRE-PROCESSAMENTO
df = pd.DataFrame([venda])

df["days_to_black_friday"] = (pd.to_datetime(df["purchase_date"]) - BLACK_FRIDAY).dt.days.abs()
df = df.drop(columns=["purchase_date"])

dados_num = df[COLUNAS_NUM]
dados_num_norm = scaler.transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=COLUNAS_NUM)

dados_cat = pd.get_dummies(df[COLUNAS_CAT], prefix_sep="_", dtype=int)
dados_bin = df[COLUNAS_BIN].reset_index(drop=True)

atributos = dados_num_norm.join(dados_cat.reset_index(drop=True)).join(dados_bin)
atributos = atributos.reindex(columns=colunas, fill_value=0)

# /\/\ INFERENCIA
def predizer(nome, modelo, atributos):
    classe  = modelo.predict(atributos)[0]
    proba = modelo.predict_proba(atributos)[0]
    classes = modelo.classes_
    confianca = proba.max() * 100

    print(f"{nome}\n")
    print(f"  Predição: {classe}")
    print(f"  Confiança: {confianca:.2f}%")
    print(f"  Distribuição de probabilidade:")
    for c, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        print(f"    {c:<20} {p*100:.2f}%")

print("\nRESULTADO DA ANÁLISE DE VENDA")

predizer("Categoria do Produto: ", modelo_category, atributos)
predizer("Forma de Pagamento: ", modelo_payment,  atributos)
predizer("Faixa Etária: ", modelo_age, atributos)

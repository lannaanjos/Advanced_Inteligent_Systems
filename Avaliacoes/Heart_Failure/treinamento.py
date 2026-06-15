## QUESTÃO 1 (1,0 ponto)
# Heart Failure
# https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
# Os dados contem informações sobre pacientes de cardiologia.
# Tarefa: Treinar um sistema inteligente que receba os dados de um paciente desconhecido e indique a qual grupo esse paciente pertente (ou é similar).
# Você deve:
#        a)       Justificar a escolhar do metaestimador
#        b)       Demonstrar todos os procedimentos de pré processamento
#        c)       Demonstrar a inferência funcionando
#
# Atenção: A base possui informações binárias que requerem que você analise melhor como essas informações serão tratadas. 

# libs utils 
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path(__file__).parent.parent.parent / "Data" / "heart_failure_clinical_records_dataset.csv"
NORM_PATH = Path(__file__).parent / "norm_heart_failure.pkl"
CLUSTER_PATH = Path(__file__).parent / "cluster_heart_failure.pkl"
COLS_PATH = Path(__file__).parent / "nomes_cols.pkl"
ELBOW_PATH = Path(__file__).parent / "cotovelo.png"

# cols q n precisam de encoding: anaemia, diabetes, high blood pressure, sex, smoking, DEATH EVENT
# cols q precisam de norm: age, creatine phosphokinase, ejection fraction, platelets, serum creatinine, serum sodium, time

COLUNAS_CONT = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
COLUNAS_BIN = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

K_MAX = 30 # dataset tem 299 linhas, k_max <= sqrt(n/2)
# o certo era 12 mas 30 é mais confortável


# agr modular :)
# UTILS

# /\/\ NORMALIZAÇÂO E PREPROCESSAMENTO
def preprocessamento(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = df.drop(columns=["DEATH_EVENT"])
 
    dados_cont = df[COLUNAS_CONT]
    dados_bin  = df[COLUNAS_BIN]
 
    scaler = MinMaxScaler()
    dados_cont_norm = scaler.fit_transform(dados_cont)
    dados_cont_norm = pd.DataFrame(dados_cont_norm, columns=COLUNAS_CONT)
 
    dados_norm = dados_cont_norm.join(dados_bin.reset_index(drop=True))
 
    pickle.dump(scaler, open(NORM_PATH, "wb"))
    print(f"Scaler salvo em {NORM_PATH}")
    print(f"Shape após pre-processamento: {dados_norm.shape}")


    return dados_norm, scaler

    
def encontrar_k_otimo(dados: pd.DataFrame) -> int:
    K = range(1, K_MAX + 1)
    distorcoes = []
 
    for k in K:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(dados)
        distorcoes.append(
            sum(np.min(cdist(dados, modelo.cluster_centers_, "euclidean"), axis=1))
            / dados.shape[0]
        )
 
    x0, y0 = K[0],  distorcoes[0]
    xn, yn = K[-1], distorcoes[-1]
 
    distancias = []
    for x, y in zip(K, distorcoes):
        numerador   = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
        denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
        distancias.append(numerador / denominador)
 
    k_otimo = K[distancias.index(max(distancias))]
 
    plt.figure()
    plt.plot(list(K), distorcoes, marker="o", markersize=4)
    plt.axvline(x=k_otimo, color="red", linestyle="--", label=f"k otimo = {k_otimo}")
    plt.title("Curva do Cotovelo -- Heart Failure")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Distorção média")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ELBOW_PATH)
    plt.close()
 
    print(f"K ótimo: {k_otimo}")
    print(f"Curva salva em: {ELBOW_PATH}")
 
    return k_otimo
 
 
def treinar(dados: pd.DataFrame, k: int) -> KMeans:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(dados)
 
    pickle.dump(modelo, open(CLUSTER_PATH, "wb"))
    pickle.dump(dados.columns.tolist(), open(COLS_PATH, "wb"))
 
    print(f"Modelo salvo em: {CLUSTER_PATH}")
    print(f"Colunas salvas em: {COLS_PATH}")
 
    return modelo
 
 
def main():
    df = pd.read_csv(DATA_PATH, sep=",")
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
 
    dados_prontos, _ = preprocessamento(df)
    k_otimo = encontrar_k_otimo(dados_prontos)
    modelo = treinar(dados_prontos, k_otimo)
 
    print(f"\nTreinamento concluído. Clusters: {k_otimo}")
    print(f"Tamanho dos clusters:\n{pd.Series(modelo.labels_).value_counts().sort_index()}")
 
 
if __name__ == "__main__":
    main()



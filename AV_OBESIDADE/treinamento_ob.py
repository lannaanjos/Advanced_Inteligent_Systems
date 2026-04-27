#907543
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from pathlib import Path

import pickle 
import numpy as np
import pandas as pd 

import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

DADOS_OBESIDADE = Path(__file__).parent / "ObesityDataSet_raw_and_data_sinthetic.csv" 
NORM_PATH = Path(__file__).parent / "norm_obesidade.pkl"
CLUSTER_DESTINO_PATH = Path(__file__).parent / "obesidade_cluster.pkl"
MAIS_COL_PATH = Path(__file__).parent / "col_names.pkl"

dados_ob = pd.read_csv(DADOS_OBESIDADE, sep=',')

dados_num = dados_ob.drop(columns=["Gender", "family_history_with_overweight", "FAVC", "CAEC",
                                   "SMOKE", "SCC", "CALC","MTRANS", "NObeyesdad"])

dados_cat = dados_ob.drop(columns=["Age", "Height", "Weight", "FCVC","NCP", "CH2O", "FAF", "TUE"])


scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

pickle.dump(normalizador, open(NORM_PATH, "wb"))

dados_num_norm = normalizador.fit_transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

dados_norm = dados_num_norm.join(dados_cat_norm, how='left')
print(dados_norm.head())

# hiperparametrização

distorcoes = []

K=range(1, 500) # meu computador n aguenta as 2000 e poucas
for i in K:
    cluster_obesidade = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    distorcoes.append(sum(
        np.min(cdist(
            dados_norm, cluster_obesidade.cluster_centers_, 'euclidean' 
        ), axis=1) / dados_norm.shape[0]
    ))

# n otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)

    distancias.append(numerador/denominador)

n_otimo_clusters = K[distancias.index(np.max(distancias))]

print(f"N° ótimo de clusters: {n_otimo_clusters}")

# treinando e salvando :D 
cluster_obesidade = KMeans(n_clusters=n_otimo_clusters, random_state=42).fit(dados_norm)

pickle.dump(cluster_obesidade, open(CLUSTER_DESTINO_PATH, 'wb'))
# salvando as cols do one hot enconding
pickle.dump(dados_norm.columns.tolist(), open(MAIS_COL_PATH, 'wb'))




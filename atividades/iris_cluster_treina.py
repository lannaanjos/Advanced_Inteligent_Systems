import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.cluster import KMeans # KMenas é um meta estimdor
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np

from pathlib import Path

CAMINHO_IRIS = Path(__file__).parent.parent / "Dados" / "iris.csv"

dados = pd.read_csv(CAMINHO_IRIS, sep=';')

# separação de dados numéricos e categóricos
dados_numericos = dados.drop(columns=['class'])
dados_categoricos = dados['class']

# normalização dados numéricos
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_numericos)

# salvar normalizador
pickle.dump(normalizador, open("normalizador_iris.pkl", "wb"))

# normalizar dados num
dados_num_norm = normalizador.fit_transform(dados_numericos)

dados_cat_norm = pd.get_dummies(
    dados_categoricos,
    prefix_sep='_',
    dtype=int
)

# transforma num norm em dataframe
dados_num_norm = pd.DataFrame(
    dados_num_norm, columns=dados_numericos.columns
)

# recompor dataframe com todos os dados
dados_norms = dados_num_norm.join(dados_cat_norm, how='left')
print(dados_norms.head(8))

# hiperparametrizar antes de treinar
distorcoes = []

# criar um intervalo numérico fechado a squerda e aberto a direita
K = range(1,101)

for i in K:
    # treina iterativamente a aumenta nro de clusters
    cluster_iris = KMeans(
        n_clusters=i,
        random_state=42
    ).fit(dados_norms)

    # calcula distorcao
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_norms, cluster_iris.cluster_centers_, 'euclidean'),axis=1) / dados_norms.shape[0]
        )
    )

# plotagem do gráfico de distorções (copiar do professor dps)
    

# determinar numero ótimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]

distancias = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]

    numerador = abs()
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


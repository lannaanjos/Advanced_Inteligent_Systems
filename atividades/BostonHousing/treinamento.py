'''
Instruções

    Utilize o arquivo HousingData.csv disponível na pasta da Aula6
    A descrição do arquivo pode ser encontrada em https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
    Treine um modelo de clusters conforme nossas aulas
    Implemente um módulo de descrição dos segmentos obtidos durante o treinamento
    Implemente um módulo de inferência que receba os dados de um imóvel desconhecido e informe a qual cluster tal imóvel pertence
    '''
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle 
from pathlib import Path
import numpy as np
import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

DADOS_BOSTON = Path(__file__).parent / "HousingData.csv"
DESTINO_NORM = Path(__file__).parent / "norm_boston_housing.pkl"
DESTINO_CLUSTER = Path(__file__).parent / "cluster_boston_housing.pkl"

dados = pd.read_csv(DADOS_BOSTON) # vou usar assim porque não há dados categóricos 
dados = dados.fillna(dados.mean()) # tratando 
scaler = MinMaxScaler()
normalizador = scaler.fit(dados)

# salvando normalizador
pickle.dump(normalizador, open(DESTINO_NORM, "wb"))

dados_num = normalizador.fit_transform(dados)

dados_num = pd.DataFrame(
    dados_num, columns=dados.columns
)

# hiperparametrização
distorcoes = []
# pega o total da base 
k_range = range(1, 506)
for k in k_range:
    # treina cada vez aumenta o número de clusters 
    cluster_boston = KMeans(
    n_clusters = k,
    random_state = 42 
    ).fit(dados_num)

    distorcoes.append(sum(np.min(
        cdist(dados_num, cluster_boston.cluster_centers_, "euclidean"), axis = 1) / dados_num.shape[0]
    ))

'''fig, ax = plt.subplots()
ax.plot(k_range, distorcoes)
ax.set_xlabel("N_clusters")
ax.set_ylabel("Distorção")
ax.set_title("Cotovelo")
ax.grid()
plt.show()'''

# determinando nº ótimo de clusters
x0 = k_range[0]
y0 = distorcoes[0]
xn = k_range[-1]
yn = distorcoes[-1]

distancias = []

for i in range(len(distorcoes)):
    x = k_range[i]
    y = distorcoes[i]

    numerador = abs(
        (yn - y0)*x - (xn-x0)*y + xn*y0 - yn*x0 
    )

    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )

    distancias.append(numerador/denominador)

numero_otimo = k_range[distancias.index(np.max(distancias))]
print(f"Número ótimo de clusters: {numero_otimo}")

# treinar e salvar modelo
cluster_boston = KMeans(
    n_clusters=numero_otimo,
    random_state=42
).fit(dados_num)

pickle.dump(cluster_boston, open(DESTINO_CLUSTER, 'wb'))




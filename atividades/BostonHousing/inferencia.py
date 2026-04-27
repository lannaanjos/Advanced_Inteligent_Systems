'''
Instruções

Utilize o arquivo HousingData.csv disponível na pasta da Aula6
A descrição do arquivo pode ser encontrada em https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
Treine um modelo de clusters conforme nossas aulas
Implemente um módulo de descrição dos segmentos obtidos durante o treinamento
Implemente um módulo de inferência que receba os dados de um imóvel desconhecido e informe a qual cluster tal imóvel pertence
'''
import pickle
import pandas as pd
from pathlib import Path

PATH_NORMALIZADOR = Path(__file__).parent / 'norm_boston_housing.pkl'
PATH_CLUSTERS = Path(__file__).parent / 'cluster_boston_housing.pkl'

colunas = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT' 
]

# criar um df vazio com a estrutura dos centroides
bh_norm = pd.DataFrame(columns = colunas)

nova_housing_data = pd.DataFrame([[5.1, 0.4, 20, 0, 0.311, 7.2, 81.7, 7.01, 4, 19.7, 390.78, 16.3, 13.9]], columns=colunas)

# norm nova entrada 
# load norm, salvo no treinamento
normalizador = pickle.load(open(PATH_NORMALIZADOR, 'rb'))
norm_nhd = normalizador.transform(nova_housing_data)
norm_nhd = pd.DataFrame(norm_nhd, columns=colunas)

# inferir o cluster pertencente
# carregar modelo de clusters
cluster_bh = pickle.load(open(PATH_CLUSTERS, 'rb'))
cluster_predicao = cluster_bh.predict(norm_nhd)
print('Cluster da nova entrada: ', cluster_predicao)


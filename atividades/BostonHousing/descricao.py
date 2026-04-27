import pickle 
import pandas as pd 
from pathlib import Path 

ORIGEM_PICKLE_BOSTON = Path(__file__).parent / "cluster_boston_housing.pkl"
ORIGEM_NORM_BOSTON = Path(__file__).parent / "norm_boston_housing.pkl"

colunas = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' 
]

# abrindo modelo
cluster_boston = pickle.load(open(ORIGEM_PICKLE_BOSTON, 'rb'))

# converter centroides em df 
centroides = pd.DataFrame(cluster_boston.cluster_centers_, columns=colunas)

# carregar normalizador
normalizador = pickle.load(open(ORIGEM_NORM_BOSTON, 'rb'))
dados_numericos = normalizador.inverse_transform(centroides)

# tem q recriar o DataFrame
dados_numericos = pd.DataFrame(dados_numericos, columns=colunas)

# n precisa desnorm cols categoricas nem juntas os dfs pq n têm dados categoricos no csv de origm
print(dados_numericos)

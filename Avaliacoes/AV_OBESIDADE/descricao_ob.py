import pickle
import pandas as pd
from pathlib import Path

CLUSTER_DESTINO_PATH = Path(__file__).parent / "obesidade_cluster.pkl"
NORM_PATH = Path(__file__).parent / "norm_obesidade.pkl"
MAIS_COL_PATH = Path(__file__).parent / "col_names.pkl"

# carrega colunas do hot encoded 
labels_colunas = pickle.load(open(MAIS_COL_PATH, 'rb'))

# separa colunas
colunas_num = ["Age", "Height", "Weight", "FCVC","NCP", "CH2O", "FAF", "TUE"]

cluster_obesidade = pickle.load(open(CLUSTER_DESTINO_PATH, 'rb'))
centroides = pd.DataFrame(cluster_obesidade.cluster_centers_, columns=labels_colunas)

colunas_cat = [col for col in centroides.columns if col not in colunas_num]

# pega cada categoria
dados_num_norm = centroides[colunas_num]

dados_cat_norm = centroides[colunas_cat]

# inverse   
normalizador = pickle.load(open(NORM_PATH, 'rb'))
dados_num = normalizador.inverse_transform(dados_num_norm)

# pega dados originais
dados_num = pd.DataFrame(dados_num, columns=dados_num_norm.columns)

dados_cat = pd.get_dummies(dados_cat_norm.round(0).astype(int))

dados_cat.columns = colunas_cat

dados_cluster_ob = dados_num.join(dados_cat)

print(dados_cluster_ob)





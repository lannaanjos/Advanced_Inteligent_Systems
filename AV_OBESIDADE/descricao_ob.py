import pickle
import pandas as pd
from pathlib import Path

CLUSTER_DESTINO_PATH = Path(__file__).parent / "obesidade_cluster.pkl"
NORM_PATH = Path(__file__).parent / "norm_obesidade.pkl"



labels_colunas = ["Gender","Age","Height","Weight","family_history_with_overweight",
                  "FAVC","FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"]

cluster_obesidade = pickle.load(open(CLUSTER_DESTINO_PATH, 'rb'))

centroides = pd.DataFrame(cluster_obesidade.cluster_centers_, columns=labels_colunas)

dados_num_norm = centroides.drop(columns=["Gender", "family_history_with_overweight", "FAVC", "CAEC",
                                   "SMOKE", "SCC", "CALC","MTRANS"])

dados_cat_norm = centroides[["Gender", "family_history_with_overweight", "FAVC", "CAEC",
                                   "SMOKE", "SCC", "CALC","MTRANS"]]

normalizador = pickle.load(open(NORM_PATH, 'rb'))
dados_num = normalizador.inverse_transform(centroides)

dados_num = pd.DataFrame(dados_num, dados_num_norm.columns)

dados_cat = pd.get_dummies(dados_cat_norm.round(0).astype(int))

dados_cat.columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC",
                                   "SMOKE", "SCC", "CALC","MTRANS"]

dados_cluster_ob = dados_num.join(dados_cat)

print(dados_cluster_ob)





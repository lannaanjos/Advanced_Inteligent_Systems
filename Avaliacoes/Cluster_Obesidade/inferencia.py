import pickle
import pandas as pd
from pathlib import Path

colunas = ["Gender","Age","Height","Weight","family_history_with_overweight","FAVC",
           "FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"]

NORM_PATH = Path(__file__).parent / "norm_obesidade.pkl"
CLUSTER_DESTINO_PATH = Path(__file__).parent / "obesidade_cluster.pkl"
MAIS_COL_PATH = Path(__file__).parent / "col_names.pkl"

cols_hot_enc = pickle.load(open(MAIS_COL_PATH, 'rb'))

ref = pd.DataFrame(columns=cols_hot_enc)

# novo indiviuo p testar
novo_individuo = pd.DataFrame([[27, 1.90, 54, 2, 3, 2, 2, 1]],
                               columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

#norm 
normalizador = pickle.load(open(NORM_PATH, "rb"))
novo_individuo_norm = normalizador.transform(novo_individuo)
novo_individuo_norm = pd.DataFrame(novo_individuo_norm, 
                                   columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

indiviuo_somado = pd.concat([novo_individuo_norm, ref], axis=0).fillna(0)

cluster_obesidade = pickle.load(open(CLUSTER_DESTINO_PATH, "rb"))
cluster_predicao = cluster_obesidade.predict(indiviuo_somado)
print(f"Predição: {cluster_predicao[0]}")



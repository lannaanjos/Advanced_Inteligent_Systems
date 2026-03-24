import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle

dados = pd.read_csv("./iris.csv", sep=';')

print(dados.head(10))

# separar atributos numericos e categoricos
dados_num = dados.drop(columns='class')
dados_cat = dados['class']

print(f"aiaiaia{dados_num}")

# nromalizar numericos
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)
# salvar modelo normalizador
pickle.dump(normalizador, open('normalizador_iris.pkl', 'wb'))

dados_num_norm = normalizador.fit_transform(dados_num)
dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep="_", dtype=int)
print(dados_cat_norm.head(10))

# transforma em df
dados_cat_norm = pd.DataFrame(dados_cat_norm, columns = dados_num.columns)


#hiperparametrizar antes do treinamento



import pickle
import pandas as pd

nome_colunas = ['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width', 'Iris-setosa',
                 'Iris-versicolor', 'Iris-virginica']

# abrir o modelo treinado
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

#imprimir valores centroides
print(cluster_iris.cluster_centers_)

# converter centroides em dataframe
centroides = pd.DataFrame(cluster_iris.cluster_centers_, columns=nome_colunas)

#print(centroides)

# segmentar dataframe em colunas numericas e colunas categoricas
dados_num_norm = centroides.drop(columns=['Iris-setosa', 'Iris-versicolor', 'Iris-viriginica'])

# desnormalizar colunas numéricas
## carregar o normalizador salvo durante preprocessamento

normalizador = pickle.load(open('normalizador_iris.pkl', 'rb'))
dados_num = normalizador.inverse_transform(dados_num_norm)

# ATENÇÃO:::::
# após desormalizador dados numéricos teremos uma matriz do numpy, será necessário recriar o df

dados_num = pd.DataFrame(dados_num, column = dados_num_norm.columns)
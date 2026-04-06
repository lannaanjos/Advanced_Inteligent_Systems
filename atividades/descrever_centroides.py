import pickle

nome_colunas = ['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width', 'Iris-setosa',
                 'Iris-versicolor', 'Iris-virginica']

# abrir o modelo treinado
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

#imprimir valores centroides
print(cluster_iris.cluster_centers_)
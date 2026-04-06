import pickle

# abrir o modelo treinado
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

#imprimir valores centroides
print(cluster_iris.cluster_centers_)
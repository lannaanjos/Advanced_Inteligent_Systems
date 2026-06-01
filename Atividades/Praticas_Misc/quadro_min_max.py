import numpy as np
from sklearn.preprocessing import MinMaxScaler

dados = np.array([
    [2100],[3000],[6700],[1200]
])

scaler = MinMaxScaler(feature_range=(0,1))

dados_normalizados = scaler.fit_transform(dados)

print(f"Dados originais:\n{dados}")
print()
print(f"Dados nornalizados:\n{dados_normalizados}")
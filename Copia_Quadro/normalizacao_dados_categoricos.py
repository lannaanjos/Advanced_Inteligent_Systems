import numpy as np
from sklearn.preprocessing import LabelEncoder

cores = ["Vermelho", "Branco", "Azul", "Roxo"]

encoder = LabelEncoder()

cored_codificadas = encoder.fit_transform(cores)
print(cored_codificadas)
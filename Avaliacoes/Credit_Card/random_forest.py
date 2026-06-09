# Utilitários
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Visualização
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_CM_RF = Path(__file__).parent / "confusion_matrix_rf.png"
PATH_CREDITO_RF = Path(__file__).parent / "credito_rf.pkl"
PATH_SPLITS = Path(__file__).parent

att_treino = pickle.load(open(PATH_SPLITS / "att_treino.pkl", "rb"))
att_teste = pickle.load(open(PATH_SPLITS /"att_teste.pkl", "rb"))
classe_treino = pickle.load(open(PATH_SPLITS / "classe_treino.pkl", "rb"))
classe_teste = pickle.load(open(PATH_SPLITS / "classe_teste.pkl", "rb"))

# HIPERPARAMETRIZAÇÃO RANDOM FOREST
# Definição dos domínios para os hiperparametros + criação da grade de valores
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)], 
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=2)],
    'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

rf_search = RandomizedSearchCV(
    estimator = rf,
    param_distributions = rf_grid,
    n_iter = 10,
    cv = 3,
    verbose = 2,
    n_jobs = -1,
    random_state = 42
)

rf_search.fit(att_treino, classe_treino)

# Mostra resultado da hiperparametrização
print("Melhores parâmtros Random Forest:")
print(rf_search.best_params_)

# Avaliação
melhor_rf = rf_search.best_estimator_ # pega modelo campeão da busca antes de avaliar
preditos_rf = melhor_rf.predict(att_teste)

acuracia_rf = accuracy_score(classe_teste, preditos_rf)
tn, fp, fn, tp = confusion_matrix(classe_teste, preditos_rf).ravel()
especificidade_rf = tn / (tn + fp)
sensibilidade_rf = tp / (tp + fn)

print("Acurácia RF: ", acuracia_rf)
print("Especificidade RF: ", especificidade_rf)
print("Sensibilidade RF: ", sensibilidade_rf)

ConfusionMatrixDisplay.from_estimator(melhor_rf, att_teste, classe_teste)
plt.title("Random Forest")
plt.savefig(PATH_CM_RF)
plt.close()

pickle.dump(melhor_rf, open(PATH_CREDITO_RF, "wb"))

# results:
'''Melhores parâmtros Random Forest:
{'n_estimators': 70, 'min_samples_split': 2, 'max_features': 'log2', 'max_depth': 62, 'criterion': 'gini'}
Acurácia RF:  0.8728867964904772
Especificidade RF:  0.9083488472003437
Sensibilidade RF:  0.8376918703808982'''

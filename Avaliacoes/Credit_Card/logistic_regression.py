# Utilitários
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Visualização
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_CM_LOGI = Path(__file__).parent / "confusion_matrix_logi.png"
PATH_CREDITO_LOGI = Path(__file__).parent / "credito_logi.pkl"
PATH_SPLITS = Path(__file__).parent

att_treino = pickle.load(open(PATH_SPLITS / "att_treino.pkl", "rb"))
att_teste = pickle.load(open(PATH_SPLITS /"att_teste.pkl", "rb"))
classe_treino = pickle.load(open(PATH_SPLITS / "classe_treino.pkl", "rb"))
classe_teste = pickle.load(open(PATH_SPLITS / "classe_teste.pkl", "rb"))

# HIPERPARAMETROS ETC ETC ETC

lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.0, 0.5, 1.0],  # 0.0 = l2, 1.0 = l1, meio-termo = elasticnet
    'solver': ['saga'],            # saga é o único que suporta l1_ratio
    'max_iter': [200, 500, 1000]
}

lr = LogisticRegression(random_state = 42)

lr_search = RandomizedSearchCV(
    estimator = lr,
    param_distributions = lr_grid,
    n_iter = 10,
    cv = 3,
    verbose = 2,
    n_jobs = -1,
    random_state = 42
)

lr_search.fit(att_treino, classe_treino)

# Mostra resultado da hiperparametrização
print("Melhores parâmetros Logistic Regression:")
print(lr_search.best_params_)

# Avaliação
melhor_lr = lr_search.best_estimator_ # pega modelo campeão da busca antes de avaliar
preditos_lr = melhor_lr.predict(att_teste)

acuracia_lr = accuracy_score(classe_teste, preditos_lr)
tn, fp, fn, tp = confusion_matrix(classe_teste, preditos_lr).ravel()
especificidade_lr = tn / (tn + fp)
sensibilidade_lr = tp / (tp + fn)

print("Acurácia Logistic Regression: ", acuracia_lr)
print("Especificidade Logistic Regression: ", especificidade_lr)
print("Sensibilidade Logistic Regression: ", sensibilidade_lr)

ConfusionMatrixDisplay.from_estimator(melhor_lr, att_teste, classe_teste)
plt.title("LogisticRegression")
plt.savefig(PATH_CM_LOGI)
plt.close()

'''Melhores parâmetros Logistic Regression:
{'solver': 'saga', 'max_iter': 500, 'l1_ratio': 1.0, 'C': 1}
Acurácia Logistic Regression:  0.6066766531136315
Especificidade Logistic Regression:  0.5834168695403121
Sensibilidade Logistic Regression:  0.6297612279704378
'''


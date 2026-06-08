# Treine e avalie a acurácia de 3 modelos
# Random Forests
# Support Vector Machine
# Um classificado a sua escolha
# Compare a acurácia detalhadamente e indique qual modelo é mais adequado para entrar em produção

# Link para a base: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Dica do Prof Escobar: não usar split 70/30 se for usar cross validation

# Utilitários
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Visualização
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib # estou fazendo no Linux, aí precisa fazer isso para poder mostrar o result do matplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_CM_RF = Path(__file__).parent / "confusion_matrix_rf.png"

# Synthetic Minority Over-sampling Technique
# -> Técnica de pré-processamento usada para resolver o desequilíbrio de um dataset. Etapas:
# 1. Pega a diferença entre uma sample e seu vizinho mais próximo.
# 2. Multiplica a diferença por um número aleatório entre 1 e 0.
# 3. Adiciona essa diferença à sample para gerar um novo exemplo sintetico no espaço de features
# 4. Continua com o próximo vizinho mais próximo até uma range definida pelo user.
from imblearn.over_sampling import SMOTE

# /\ CARREGANDO ARQUIVOS
DADOS_PATH = Path(__file__).parent.parent.parent / "Data" / "diabetes.csv"
DIABETES_TREE_PATH = Path(__file__).parent / "diabetes_tree.pkl"

dados = pd.read_csv(DADOS_PATH, sep=',')

# verificando se pegou certo
# print(dados)

atributos = dados.drop(columns=["Outcome"])
classe = dados["Outcome"]

# /\ Balanceamento dos Dados
print("Sequência das classes:", dados["Outcome"].value_counts())
balancer = SMOTE()
atributos, classe = balancer.fit_resample(atributos, classe)

# /\ Segmentação dos dados de treino e dados de teste
att_treino, att_teste, classe_treino, classe_teste = train_test_split(atributos, classe, test_size=0.3)

tree = DecisionTreeClassifier(random_state=42)

# Treinamento do modelo
diabetes_tree = tree.fit(att_treino, classe_treino)

# Salvando modelo
pickle.dump(diabetes_tree, open(DIABETES_TREE_PATH, "wb"))

# HIPERPARAMETRIZAÇÃO RANDOM FOREST
# Definição dos domínios para os hiperparametros + criação da grade de valores
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    # Criterion é  afunção usada p/ medir a qualidade de uma divisão na árvore interna da RF
    # - Gini: mede impureza, ou seja, o quão misturadas as classes estão
    # - Entropy: mede ganho de informação, ou seja, o quanto uma divisão reduz a incerteza 
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


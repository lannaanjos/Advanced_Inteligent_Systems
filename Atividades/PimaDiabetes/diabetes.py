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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Visualização
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Synthetic Minority Over-sampling Technique
# -> Técnica de pré-processamento usada para resolver o desequilíbrio de um dataset. Etapas:
# 1. Pega a diferença entre uma sample e seu vizinho mais próximo.
# 2. Multiplica a diferença por um número aleatório entre 1 e 0.
# 3. Adiciona essa diferença à sample para gerar um novo exemplo sintetico no espaço de features
# 4. Continua com o próximo vizinho mais próximo até uma range definida pelo user.
from imblearn.over_sampling import SMOTE

from pathlib import Path

DADOS_PATH = Path(__file__).parent.parent.parent / "Data" / "diabetes.csv"
DIABETES_TREE_PATH = Path(__file__).parent / "diabetes_tree.pkl"

dados = pd.read_csv(DADOS_PATH, sep=',')

# verificando se pegou certo
# print(dados)

atributos = dados.drop(columns=["Outcome"])
classe = dados["Outcome"]

# Balanceamento
print("Sequência das classes:", dados["Outcome"].value_counts())

balancer = SMOTE()
atributos, classe = balancer.fit_resample(atributos, classe)

# Separando dados de treino de dado de teste
att_treino, att_teste, classe_treino, classe_teste = train_test_split(atributos, classe, test_size=0.3)

tree = DecisionTreeClassifier(random_state=42)

# Treinamento
diabetes_tree = tree.fit(att_treino, classe_treino)

# salvando
pickle.dump(diabetes_tree, open(DIABETES_TREE_PATH, "wb"))

preditos = diabetes_tree.predict(att_teste)
# print(preditos)

ConfusionMatrixDisplay.from_estimator(diabetes_tree, att_teste, classe_teste)
plt.show()

# Acurácia Geral
acuracia = accuracy_score(classe_teste, preditos)
print("Acurácia: ", acuracia)

tn, fp, fn, tp = confusion_matrix(classe_teste, preditos).ravel()

# especificidade
especificidade = tn / (tn + fp)

print("Especificidade: ", especificidade)

# sensibilidade
sensibilidade = tp / (tp + fn)

print("Sensibilidade: ", sensibilidade)
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE

from pathlib import Path

DADOS_PATH = Path(__file__).parent.parent.parent / "Data" / "diabetes.csv"

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
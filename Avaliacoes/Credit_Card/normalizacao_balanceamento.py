# O Banco AgoraVai precisa detereminar se os clientes do seu cartão de crédito se tornarão inadimplementes ou não (em ingles Default).
# 
# Para tanto, você deve implementar um sistema inteligente que determine se um cliente tem risco e qual é o risco (score) de se tornar inadimplente usando o arquivo de dados default_of_credit_card_clients.
# 
# Critérios de avaliação:
#
#    Pipeline completo
#    Seleção do melhor modelo
#    Módulo de inferência
#    Resultado acompanhado da distribuição de probabilidade
#
#
# Entregue o link para o seu repositório de código.

# TARGET = default payment next month
# segundo o kaggle não tem nenhum colunas com dados missing
# prof falou que é bom olhar o dataset mas ele parece estar ok
# aparentemente há dados q parecem num mas são cat

# UTIL
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# VISU
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_CM_RF = Path(__file__).parent / "confusion_rf.png" # random forest
PATH_CM_DT = Path(__file__).parent / "confusion_dt.png" # decision tree

ATT_TREINO_PATH = Path(__file__).parent / "att_treino.pkl"
ATT_TESTE_PATH = Path(__file__).parent / "att_teste.pkl"
CLASSE_TREINO_PATH = Path(__file__).parent / "classe_treino.pkl"
CLASSE_TESTE_PATH = Path(__file__).parent / "classe_teste.pkl"

PATH_CREDITO_TREE = Path(__file__).parent / "credito_tree.pkl"

# /\/\ CARREGANDO ARQUIVO
DADOS_PATH = Path(__file__).parent.parent.parent / "Data" / "default_of_credit_card_clients.csv"
dados = pd.read_csv(DADOS_PATH, sep=';')

print(dados)

# /\/\ NORMALIZAÇÃO DOS DADOS
# Separar categoricos de numericos
dados.drop(columns=["ID"])

dados_enc = pd.get_dummies(dados, columns = ['SEX', 'EDUCATION', 'MARRIAGE'])

classe = dados_enc['default payment next month']
atributos = dados_enc.drop(columns = ['default payment next month'])
pickle.dump(list(atributos.columns), open(Path(__file__).parent / "colunas.pkl", "wb"))

# /\/\ BALANCEAMENTO DOS DADOS
balancer = SMOTE()

atributos, classe = balancer.fit_resample(atributos, classe)


# SEPARAÇÃO DADOS DE TREINO E DADOS DE TESTE
att_treino, att_teste, classe_treino, classe_teste = train_test_split(atributos, classe, test_size=0.3)


# TREINAR E SALVAR MODEL
tree = DecisionTreeClassifier(random_state = 42)
credito_tree = tree.fit(att_treino, classe_treino)

pickle.dump(credito_tree, open(PATH_CREDITO_TREE, "wb"))

# Modelos e inferência em arquivos a parte
# Seria melhor deixar tudo modular mas não deu tempo :P

# dumpando dados p trafegar entre arquivos
pickle.dump(att_treino, open(ATT_TREINO_PATH, "wb"))
pickle.dump(att_teste, open(ATT_TESTE_PATH, "wb"))
pickle.dump(classe_treino, open(CLASSE_TREINO_PATH, "wb"))
pickle.dump(classe_teste, open(CLASSE_TESTE_PATH, "wb"))




